# thesis_draft_v1.docx -> BHT LaTeX template

Converted with Pandoc + post-processing; last regenerated 28 Aug 2026 from
the updated draft (25.2k words, 22 figures, 14 tables, 6 algorithm blocks,
appendices A-D).

The conversion scripts live outside this folder; re-running them rewrites
`kapitel*/chN.tex`, `anhang.tex`, `bhtThesis.bib` and `pictures/image*.png`
only. `main.tex`, `titelseiten.tex`, `titlepage-en.tex` and
`acknowledgements.tex` are hand-maintained and are never overwritten.
Source: `../thesis_draft_v1.docx`, template: `~/Downloads/bhtThesis`.

## Build

    make            # pdflatex -> bibtex -> pdflatex x2
    # or: latexmk -pdf main.tex

## What is where

| file | content |
|---|---|
| `main.tex` | metadata (author, title, Gutachter), package setup, chapter includes |
| `kapitel1..7/chN.tex` | chapters 1-7, one file each |
| `anhang.tex` | Appendix A and B |
| `bhtThesis.bib` | 24 entries generated from the Word reference list |
| `titelseiten.tex` | title page, abstracts, Erklärung, TOC |
| `acknowledgements.tex` | **empty — text still to be written** |
| `abstract_de.tex`, `abstract_en.tex` | not included; re-enable in `titelseiten.tex` |
| `pictures/` | BHT logos + the four chapter-3 figures |

## Changes made to the university template

* `\documentclass[... english]{book}` + `\selectlanguage{english}` — the .sty
  loads `ngermanb`; the thesis is in English.
* `booktabs`, `longtable`, `xspace`, `url`, `natbib` added (needed by the
  converted tables and citations).
* `\bibliographystyle{plainnat}` instead of the supplied `myapalike`:
  myapalike is German-language ("und", "Seiten") and not natbib-compatible.
  To go back: set `myapalike`, drop natbib, and change `\citep{x}`/`\citet{x}`
  to `\cite{x}`.
* `\renewcommand{\theabschluss}{Master of Science (M.Sc.)}` — the .sty
  defaults to Bachelor of Engineering.
* `titlepage-en.tex` redefines `\bhtTitelSeiteNeu` with English wording and
  sets `\headcolor` to black. `bhtThesis.sty` itself is untouched — delete
  the `\input{titlepage-en.tex}` line in `main.tex` to get the German
  blue-headline original back.

## Digital vs. print

`main.tex` starts with two `\documentclass` lines. The **digital** one is
active (`oneside`, `\digitaltrue`): no blank verso pages, chapters start on
whatever page follows. To produce the **print** version, comment the digital
pair and uncomment the print pair (`twoside, openright`, `\digitalfalse`) —
`titelseiten.tex` then re-inserts the blank backs of the title page and the
Aufgabenblatt.

Front matter is numbered i, ii, iii...; the body restarts at 1 with Chapter 1.

Digital: 46 pages, no blank pages. Print: ~53 pages.

## TODO before submission

1. Write `acknowledgements.tex`.
2. Zusammenfassung, Abstract and the Aufgabenblatt page are commented out in
   `titelseiten.tex`. **Check with your Betreuer** — BHT normally expects the
   signed Aufgabenblatt and an abstract in the submitted version; uncomment
   the relevant block to bring them back.
3. Check every entry in `bhtThesis.bib` — it was parsed out of the plain-text
   reference list.
4. Two citations could not be resolved automatically and are still plain text:
   `(Wang et al., 2025)` / `Wang et al.'s (2025)` — ambiguous between
   `wang2025` (LinkAlign) and `wang2025b` (AutoLink).
5. `Yang (2025)` in the text has no matching reference; the list has
   Yang et al. (2024).
6. `Section 4.4.3` is referenced but does not exist (4.4 has only 4.4.1, 4.4.2).
The appendix JSON record and the three system prompts are `lstlisting`
blocks (style `thesisblock`, defined in `main.tex`) and are captioned as
Listing A.1, B.1, B.2, B.3.
