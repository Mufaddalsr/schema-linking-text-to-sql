"""Tests for ``schema_linking.utils.llm_client``.

All tests use :class:`MockLLMClient` — no real Anthropic API calls. An
autouse fixture monkeypatches ``time.sleep`` so tenacity's exponential
backoff doesn't actually block the test run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
import httpx
import pytest

from schema_linking.utils.llm_client import (
    CostCapExceeded,
    LLMResponse,
    MockLLMClient,
    MockTurn,
    _last_user_message_key,
    _prompt_hash,
)

MODEL = "claude-haiku-4-5-20251001"


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda seconds: None)


def _client(tmp_path: Path, responses: dict[str, MockTurn], **kwargs) -> MockLLMClient:
    return MockLLMClient(
        model=MODEL,
        temperature=0.0,
        max_tokens=1024,
        log_path=tmp_path / "llm_calls.jsonl",
        responses=responses,
        **kwargs,
    )


def test_basic_call_returns_sensible_response(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "hello"}]
    key = _last_user_message_key(messages)
    client = _client(tmp_path, {key: MockTurn(text="hi there", input_tokens=10, output_tokens=5)})

    response = client.call(system="be helpful", messages=messages)

    assert isinstance(response, LLMResponse)
    assert response.text == "hi there"
    assert response.input_tokens == 10
    assert response.output_tokens == 5
    assert response.cost_usd > 0
    assert response.stop_reason == "end_turn"
    assert response.latency_ms >= 0


def test_retry_recovers_after_two_failures(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "flaky"}]
    key = _last_user_message_key(messages)
    turn = MockTurn(text="ok", raises=[_rate_limit_error(), _rate_limit_error()])
    client = _client(tmp_path, {key: turn})

    response = client.call(system="s", messages=messages)

    assert response.text == "ok"


def test_retry_ceiling_raises_after_five_attempts(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "always flaky"}]
    key = _last_user_message_key(messages)
    turn = MockTurn(text="never seen", raises=[_rate_limit_error() for _ in range(5)])
    client = _client(tmp_path, {key: turn})

    with pytest.raises(anthropic.RateLimitError):
        client.call(system="s", messages=messages)


def test_cost_cap_exceeded(tmp_path: Path) -> None:
    log_path = tmp_path / "llm_calls.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "model": MODEL,
                "temperature": 0.0,
                "prompt_hash": "deadbeef",
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cost_usd": 14.90,
                "latency_ms": 100,
                "stop_reason": "end_turn",
                "metadata": {},
                "response_text_hash": "abc",
            }
        )
        + "\n"
    )
    messages = [{"role": "user", "content": "one more call"}]
    key = _last_user_message_key(messages)
    # input_tokens=200_000 * $1.00/MTok == $0.20 at MODEL's pricing.
    turn = MockTurn(text="pricey", input_tokens=200_000, output_tokens=0)
    client = MockLLMClient(
        model=MODEL,
        temperature=0.0,
        max_tokens=1024,
        log_path=log_path,
        responses={key: turn},
        cost_cap_usd=15.0,
    )

    with pytest.raises(CostCapExceeded):
        client.call(system="s", messages=messages)


def test_log_line_format(tmp_path: Path) -> None:
    messages = [{"role": "user", "content": "log me"}]
    key = _last_user_message_key(messages)
    client = _client(tmp_path, {key: MockTurn(text="logged", input_tokens=7, output_tokens=3)})

    client.call(system="s", messages=messages, metadata={"method": "forward"})

    lines = (tmp_path / "llm_calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    expected_keys = {
        "timestamp", "model", "temperature", "prompt_hash", "input_tokens",
        "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
        "cost_usd", "latency_ms", "stop_reason", "metadata", "response_text_hash",
    }
    assert set(entry.keys()) == expected_keys
    assert entry["metadata"] == {"method": "forward"}
    assert "log me" not in json.dumps(entry)
    assert "logged" not in json.dumps(entry)


def test_prompt_hash_deterministic() -> None:
    system = "system prompt"
    messages = [{"role": "user", "content": "same input"}]
    assert _prompt_hash(system, messages) == _prompt_hash(system, list(messages))
