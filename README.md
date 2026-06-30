<p align="center">
  <img width="1895" height="966" alt="Image" src="https://github.com/user-attachments/assets/b35268e8-9383-4994-bad7-92014aaa899d" />
</p>

<h1 align="center">FitHire</h1>

<p align="center">
<b>Rank candidates the way a great recruiter would.</b><br>
Not by matching keywords, but by understanding people.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Semantic%20Search-SentenceTransformers-FF6F00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Data%20%26%20AI-Challenge-orange?style=for-the-badge)

</p>

---

## Problem Statement

> Recruiters go through hundreds of profiles and still often miss the right person — not because the talent isn't there, but because keyword filters can't see what actually matters.

**The Data & AI Challenge** asked for an AI system that ranks candidates the way a great recruiter would: by reading a job description and *understanding* what the role needs, looking at the full picture (career history, skills, behavioral signals, platform activity), and delivering a shortlist a recruiter can actually trust.

**FitHire** is that system.

Instead of asking *"do these words match?"*, it asks: **"would this person actually succeed in this role?"**

---

## Demo

[▶ Watch Demo](https://github.com/user-attachments/assets/b5ee7a97-6904-4b23-8f17-b988638783df)

The demo walks through:
- Landing on a resume-style homepage that scrolls down into the upload workflow
- Uploading a Job Description (PDF / DOCX / TXT / paste / or even a photo of one)
- Uploading a candidate pool (Excel / CSV / JSON / JSONL)
- Watching the Adaptive Intelligence Scoring pipeline rank the pool
- Reviewing the explainable, recruiter-friendly score breakdown per candidate
- Comparing shortlisted candidates side by side and exporting to CSV

---

## Why FitHire Is Different

Traditional ATS tools score candidates with a single fixed formula — usually something like `Skills 50% + Experience 30% + Education 20%` — applied identically whether you're hiring a DevOps engineer or an Engineering Manager. That formula punishes good candidates for missing irrelevant fields and rewards keyword stuffing over substance.

FitHire instead runs a **JD-aware, per-candidate adaptive pipeline**:

1. It reads the JD and infers what role is actually being hired for.
2. It generates a *custom weight profile* for that specific role (a DevOps JD weights production signals and certifications heavily; a Manager JD weights leadership and career progression instead).
3. For *each individual candidate*, if a signal is unavailable (e.g. no GitHub profile), its weight is redistributed across the candidate's other available signals rather than scored as a flat zero — so nobody is punished for a field that simply doesn't apply to them.
4. It blends keyword-based search (BM25) with semantic, meaning-based matching (sentence embeddings), so "AI Engineer" and "Machine Learning Engineer" are recognized as the same thing.
5. Every score ships with a recruiter-readable explanation, not just a number.

---

## Core Features

✔ AI-powered Job Description understanding (LLM mode with heuristic regex fallback when no API key is present)

✔ Dynamic, role-aware weight generation — no two job descriptions score candidates the same way

✔ Per-candidate adaptive weight redistribution — missing data is never penalized as zero

✔ Hybrid retrieval — BM25 keyword search + sentence-transformer semantic similarity + cross-encoder re-ranking

✔ Three-layer scoring model — **Technical Fit**, **Career Fit**, **Recruiter Fit**

✔ Career progression analysis — tenure stability, job-hopping detection, promotion trajectory, leadership and architecture-ownership signals

✔ Education scoring against JD requirements (penalizes under-qualification, rewards exceeding it)

✔ Built-in fairness filter — strips gender, age, religion, nationality, ethnicity, disability, and other protected attributes before any scoring happens

✔ Universal candidate normalization — accepts wildly inconsistent field names (`skills` / `technical_skills` / `tech_stack` …) and folds them into one schema

✔ OCR support — can read resumes submitted as photos/screenshots, not just structured files

✔ Explainable, recruiter-grade reasoning generated per candidate

✔ CSV export in the challenge's required output format

---

## Architecture

FitHire is two layers of the same engine: a FastAPI brain (`main.py`) that does all the actual thinking, and a thin CLI wrapper (`rank_cli.py`) that drives it for bulk, offline runs and competition-format CSV exports.

<p align="center">
  <img width="1024" height="1536" alt="Image" src="https://github.com/user-attachments/assets/55dd9f3b-9f6b-4c94-9401-d76a2284da1f" />
</p>

### `main.py` vs `rank_cli.py`

| | `main.py` | `rank_cli.py` |
|---|---|---|
| Role | The chef — owns all the intelligence | The waiter — takes the order to the chef and brings results back |
| FastAPI server | ✅ | ❌ |
| JD parsing, candidate parsing, normalization | ✅ (owns it) | Calls into `main.py` |
| Scoring & ranking logic | ✅ (owns it) | Calls into `main.py` |
| OCR | ✅ | ❌ |
| Frontend integration | ✅ | ❌ |
| Bulk processing / CSV export for submission | Limited | ✅ |

---

## Scoring Pipeline in Detail

| Stage | What happens |
|---|---|
| **Dependency loading** | Optional libraries (PyPDF2, python-docx, rank-bm25, sentence-transformers, pytesseract) are detected at startup; the system still runs with reduced capability if any are missing |
| **Fairness filter** | Protected attributes (gender, age, religion, nationality, ethnicity, disability, sexual orientation, etc.) are stripped from every candidate before scoring begins |
| **JD understanding** | AI mode extracts role title, experience range, must-have/nice-to-have skills, certifications, responsibilities, preferred companies, and soft skills; heuristic mode (regex + keyword extraction) is the automatic fallback when no AI key is configured |
| **Dynamic weight generation** | Detects the role family from the JD (e.g. DevOps vs. People Manager vs. ML Engineer) and generates a custom weight profile instead of one-size-fits-all weights |
| **Candidate normalization** | Resolves dozens of field-name synonyms into one schema, auto-detects resume sections from unstructured text, and infers years of experience from a title when it isn't stated explicitly |
| **Adaptive weight redistribution** | Per candidate, any signal with no available data has its weight redistributed across the candidate's remaining signals instead of being scored as zero |
| **Career progression analysis** | Evaluates tenure stability, job-hopping patterns, promotion trajectory, leadership language ("led", "mentored", "owned"), and architecture-ownership signals |
| **Education scoring** | Compares candidate education against JD requirements, applying a penalty multiplier for under-qualification and a bonus for exceeding it |
| **Three-layer scoring** | Combines everything into **Technical Fit**, **Career Fit**, and **Recruiter Fit** sub-scores plus a final weighted score |
| **Confidence + explanation** | Computes a confidence score and generates a recruiter-readable reasoning string for every ranked candidate |

---

## Supported Formats

| Job Description | Candidate Pool |
|---|---|
| PDF | JSON |
| DOCX | JSONL |
| TXT | CSV |
| JPG / PNG (via OCR) | Excel (.xlsx / .xls) |
| Paste directly | |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Single-page HTML/CSS/JS — resume-style landing page that scroll-reveals the upload workflow |
| Backend | FastAPI |
| Retrieval | BM25 (rank-bm25) for keyword search |
| Semantic search | sentence-transformers (`all-MiniLM-L6-v2` bi-encoder) + cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`) |
| JD/resume parsing | PyPDF2, python-docx, pandas/openpyxl, pytesseract + Pillow for OCR |
| AI reasoning | LLM-based JD analysis with a heuristic regex fallback when offline |
| Export | CSV (challenge-required format) |

---

## Project Structure

```text
FitHire
│
├── backend/
│   ├── main.py            # Ranking engine + FastAPI API server
│   ├── rank_cli.py         # CLI wrapper for bulk/offline ranking + CSV export
│   └── __init__.py
│
├── frontend/
│   └── dist/
│       └── index.html      # Resume-style landing page + upload UI
│
├── data/
│   ├── sample_jd.txt
│   ├── sample_candidates.json
│   └── sample_candidates.csv
│
├── requirements.txt
├── run.py                  # One-command launcher
└── README.md
```

---

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd FitHire

# Install dependencies
python -m pip install -r requirements.txt

# Start the server
python run.py
```

Then open:

```
http://localhost:8000
```

### Running the bulk CLI (for the competition output format)

```bash
python backend/rank_cli.py --jd data/sample_jd.txt --candidates data/sample_candidates.json --out ranked_candidates.csv
```

> Adjust the flags above to match `rank_cli.py`'s actual argument names if they differ — check `python backend/rank_cli.py --help`.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/rank` | POST | Upload a JD (file or pasted text) and a candidate pool (file or JSON text); returns the full ranked shortlist with score breakdowns |
| `/api/export/csv` | POST / GET | Export a given set of ranked results to CSV |
| `/api/health` | GET | Health check — reports which optional capabilities (BM25, transformers, OCR) are currently active |

---

## Challenge Compatibility

Built specifically for **The Data & AI Challenge**.

**Input:** a job description plus a candidate pool in any of the supported formats (including the challenge's `.jsonl` dataset).

**Output:** `ranked_candidates.csv` containing candidate ID, rank, overall score, confidence, AI-generated reasoning, matched skills, and the full Technical/Career/Recruiter Fit breakdown — ready to submit as-is.

---

## Future Improvements

- 🔹 Vector database support for million-scale candidate search
- 🔹 LLM-powered recruiter copilot ("why was this candidate selected over that one?")
- 🔹 Side-by-side candidate comparison dashboard improvements
- 🔹 ATS integrations (Greenhouse, Lever, Workday)
- 🔹 Formal fairness & bias auditing reports
- 🔹 Team collaboration and recruiter notes

---

## Built for the Data & AI Challenge

**Hiring deserves more than keyword matching.**

FitHire helps recruiters spend less time searching and more time hiring the right people.
