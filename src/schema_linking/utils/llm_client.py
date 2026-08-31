"""Anthropic API client wrapper with retries, cost tracking, and prompt caching.

Used by the LLM-based linkers (Methods C/D/E — forward, backward, bidirectional
prompting) to call Claude with automatic retry on
transient failures, a hard spend cap, and structured JSONL logging for later
cost/hallucination analysis.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

_MTOK = 1_000_000

# Anthropic API pricing in USD per million tokens (MTok).
# Source of truth: https://claude.com/pricing — re-check and update this
# table whenever a model's price changes or a new model is added. cache_write
# is the ephemeral (5-minute) cache write rate; cache_read has consistently
# been priced at 1/10th of the base input rate across Anthropic model
# generations, but verify before trusting that ratio for a new model.
_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
}


class CostCapExceeded(RuntimeError):
    """Raised when cumulative logged LLM spend has reached ``cost_cap_usd``."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Result of one :meth:`LLMClient.call`.

    Attributes
    ----------
    text
        Concatenated text content of the model's reply.
    input_tokens, output_tokens
        Non-cached token counts billed for this call.
    cache_creation_input_tokens, cache_read_input_tokens
        Tokens written to / read from the ephemeral prompt cache.
    cost_usd
        Computed spend for this call, from :data:`_PRICING_PER_MTOK`.
    latency_ms
        Wall-clock time for the (possibly retried) call to complete.
    stop_reason
        Anthropic's reported stop reason (e.g. ``"end_turn"``, ``"max_tokens"``).
    raw
        Full response payload as a dict, for ad-hoc debugging. Never logged.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    latency_ms: int
    stop_reason: str
    raw: dict[str, Any]


def _compute_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
) -> float:
    """Compute USD cost for one call from :data:`_PRICING_PER_MTOK`.

    Raises
    ------
    KeyError
        If ``model`` has no pricing entry — add one rather than guessing.
    """
    try:
        pricing = _PRICING_PER_MTOK[model]
    except KeyError as exc:
        raise KeyError(
            f"no pricing entry for model {model!r} in _PRICING_PER_MTOK "
            "(schema_linking.utils.llm_client) — add one; see https://claude.com/pricing"
        ) from exc
    cost = (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
        + cache_creation_input_tokens * pricing["cache_write"]
        + cache_read_input_tokens * pricing["cache_read"]
    ) / _MTOK
    return round(cost, 6)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_hash(system: str, messages: list[dict[str, Any]]) -> str:
    """Deterministic hash of a call's inputs, for log/dedup without storing text."""
    payload = system + json.dumps(messages, sort_keys=True)
    return _sha256(payload)


def _build_system_param(
    system: str, cacheable_prefix: str | None, use_prompt_caching: bool
) -> str | list[dict[str, Any]]:
    """Build the Anthropic ``system`` request parameter.

    When ``cacheable_prefix`` is given and caching is enabled, it becomes its
    own system block with ``cache_control: {"type": "ephemeral"}`` ahead of
    ``system``, so repeated calls sharing the same schema (same ``db_id``)
    hit the cache.
    """
    if cacheable_prefix is None:
        return system
    if not use_prompt_caching:
        return f"{cacheable_prefix}\n\n{system}"
    return [
        {"type": "text", "text": cacheable_prefix, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": system},
    ]


def _is_retryable(exc: BaseException) -> bool:
    """Connection errors and rate limits always retry; other API errors only
    for 5xx server faults. 4xx (bad request) and non-API errors (e.g. JSON
    parse failures in calling code) are never retried."""
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.RateLimitError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


def _extract_text(raw_message: Any) -> str:
    return "".join(
        block.text for block in raw_message.content if getattr(block, "type", None) == "text"
    )


def _sum_logged_cost(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    total = 0.0
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total += json.loads(line)["cost_usd"]
    return total


def _append_log_line(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class _LLMClientBase:
    """Shared retry, logging, and cost-cap logic for :class:`LLMClient` and
    :class:`MockLLMClient`. Subclasses implement only :meth:`_create_message`."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        log_path: Path,
        use_prompt_caching: bool = True,
        cost_cap_usd: float | None = 15.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_path = Path(log_path)
        self.use_prompt_caching = use_prompt_caching
        self.cost_cap_usd = cost_cap_usd

    def _create_message(
        self, system_param: str | list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Any:
        raise NotImplementedError

    def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        cacheable_prefix: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call the model, with retry, cost-cap enforcement, and JSONL logging.

        The cost cap is enforced twice: before the call (blocks further spend
        once already at/over cap) and after (a call's exact cost is only
        known once its response returns, so the call that tips the total
        over the cap still happens and is logged, but raises instead of
        returning).
        """
        prior_cost = _sum_logged_cost(self.log_path)
        if self.cost_cap_usd is not None and prior_cost >= self.cost_cap_usd:
            raise CostCapExceeded(
                f"logged spend ${prior_cost:.4f} already at/over cap ${self.cost_cap_usd:.4f}"
            )

        system_param = _build_system_param(system, cacheable_prefix, self.use_prompt_caching)
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        start = time.monotonic()
        raw_message = retrying(self._create_message, system_param, messages)
        latency_ms = int((time.monotonic() - start) * 1000)

        usage = raw_message.usage
        cost_usd = _compute_cost_usd(
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_input_tokens or 0,
            usage.cache_read_input_tokens or 0,
        )
        response = LLMResponse(
            text=_extract_text(raw_message),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            stop_reason=raw_message.stop_reason,
            raw=raw_message.model_dump(),
        )

        _append_log_line(
            self.log_path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": self.model,
                "temperature": self.temperature,
                "prompt_hash": _prompt_hash(system, messages),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cache_creation_input_tokens": response.cache_creation_input_tokens,
                "cache_read_input_tokens": response.cache_read_input_tokens,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "stop_reason": response.stop_reason,
                "metadata": metadata or {},
                "response_text_hash": _sha256(response.text),
            },
        )

        if self.cost_cap_usd is not None and prior_cost + response.cost_usd > self.cost_cap_usd:
            raise CostCapExceeded(
                f"call cost ${response.cost_usd:.4f} pushed logged spend to "
                f"${prior_cost + response.cost_usd:.4f}, over cap ${self.cost_cap_usd:.4f}"
            )

        return response


class LLMClient(_LLMClientBase):
    """Thin wrapper over :class:`anthropic.Anthropic` with retry, cost
    tracking, prompt caching, and JSONL call logging. Reads
    ``ANTHROPIC_API_KEY`` from the environment (``anthropic.Anthropic``
    default)."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        log_path: Path,
        use_prompt_caching: bool = True,
        cost_cap_usd: float | None = 15.0,
    ) -> None:
        super().__init__(
            model, temperature, max_tokens, log_path, use_prompt_caching, cost_cap_usd
        )
        self._client = anthropic.Anthropic()

    def _create_message(
        self, system_param: str | list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Any:
        return self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_param,
            messages=messages,
        )


@dataclass
class MockTurn:
    """One programmed reply for :class:`MockLLMClient`.

    Attributes
    ----------
    raises
        Exceptions to raise on the first ``len(raises)`` calls for this
        message before returning the canned response — for exercising retry
        behaviour without a real API.
    texts
        When given, successive non-raising calls to this message return
        ``texts[0]``, ``texts[1]``, ... (cycling if exhausted) instead of
        the single ``text`` value — for exercising k-samples aggregation
        (e.g. the LLM forward linker's repeated identical-prompt sampling)
        where each call must return a different canned response.
    """

    text: str = "mock response"
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    stop_reason: str = "end_turn"
    raises: list[BaseException] = field(default_factory=list)
    texts: list[str] | None = None


@dataclass
class _MockUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


@dataclass
class _MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class _MockMessage:
    content: list[_MockTextBlock]
    usage: _MockUsage
    stop_reason: str

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": [{"type": b.type, "text": b.text} for b in self.content],
            "usage": vars(self.usage),
            "stop_reason": self.stop_reason,
        }


def _last_user_message_key(messages: list[dict[str, Any]]) -> str:
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if not user_messages:
        raise ValueError("messages must include at least one user message")
    return _sha256(str(user_messages[-1]))


class MockLLMClient(_LLMClientBase):
    """Test double for :class:`LLMClient` with the same ``call()`` interface.

    Programmed with a ``responses`` mapping from the SHA-256 of a message
    list's last user turn (:func:`_last_user_message_key`) to a
    :class:`MockTurn`. Used exclusively in tests — never for real linker runs.
    """

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        log_path: Path,
        responses: dict[str, MockTurn],
        use_prompt_caching: bool = True,
        cost_cap_usd: float | None = 15.0,
    ) -> None:
        super().__init__(
            model, temperature, max_tokens, log_path, use_prompt_caching, cost_cap_usd
        )
        self._responses = responses
        self._call_counts: dict[str, int] = {}

    def _create_message(
        self, system_param: str | list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> Any:
        key = _last_user_message_key(messages)
        if key not in self._responses:
            raise KeyError(f"MockLLMClient has no programmed response for key {key!r}")
        turn = self._responses[key]
        count = self._call_counts.get(key, 0)
        self._call_counts[key] = count + 1
        if count < len(turn.raises):
            raise turn.raises[count]
        text = turn.text
        if turn.texts:
            success_index = count - len(turn.raises)
            text = turn.texts[success_index % len(turn.texts)]
        return _MockMessage(
            content=[_MockTextBlock(text=text)],
            usage=_MockUsage(
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                cache_creation_input_tokens=turn.cache_creation_input_tokens,
                cache_read_input_tokens=turn.cache_read_input_tokens,
            ),
            stop_reason=turn.stop_reason,
        )
