<p align="center">
  <img width="1895" height="966" alt="Image" src="https://github.com/user-attachments/assets/b35268e8-9383-4994-bad7-92014aaa899d" />
</p>

<h1 align="center">FitHire</h1>

<p align="center">
<b>Rank candidates the way a great recruiter would.</b><br>
Not by matching keywords, but by understanding people.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Semantic%20Search-SentenceTransformers-FF6F00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Data%20%26%20AI-Challenge-orange?style=for-the-badge)

</p>

---

## Problem Statement

> Recruiters go through hundreds of profiles and still often miss the right person - not because the talent isn't there, but because keyword filters can't see what actually matters.

**The Data & AI Challenge** asked for an AI system that ranks candidates the way a great recruiter would: by reading a job description and _understanding_ what the role needs, looking at the full picture (career history, skills, behavioral signals, platform activity), and delivering a shortlist a recruiter can actually trust.

**FitHire** is that system.

Instead of asking **"do these words match?"**, it asks: **"would this person actually succeed in this role?"**

---

## Demo

[▶ Watch Demo](https://github.com/user-attachments/assets/b5ee7a97-6904-4b23-8f17-b988638783df)

The demo walks through:

- Landing on a resume-style homepage that scrolls down into the upload workflow
- Uploading a Job Description (PDF / DOCX / TXT / paste)
- Uploading a candidate pool (Excel / CSV / JSON / JSONL)
- Watching the Adaptive Intelligence Scoring pipeline rank the pool
- Reviewing the explainable, recruiter-friendly score breakdown per candidate
- Comparing shortlisted candidates side by side and exporting to CSV

---

## Why FitHire Is Different

Traditional ATS tools score candidates with a single fixed formula - usually something like `Skills 50% + Experience 30% + Education 20%` - applied identically whether you're hiring a DevOps engineer or an Engineering Manager. That formula punishes good candidates for missing irrelevant fields and rewards keyword stuffing over substance.

FitHire instead runs a **JD-aware, per-candidate adaptive pipeline**:

1. It reads the JD and infers what role is actually being hired for.
2. It generates a _custom weight profile_ for that specific role (a DevOps JD weights production signals and certifications heavily; a Manager JD weights leadership and career progression instead).
3. For _each individual candidate_, if a signal is unavailable (e.g. no GitHub profile), its weight is redistributed across the candidate's other available signals rather than scored as a flat zero - so nobody is punished for a field that simply doesn't apply to them.
4. It blends keyword-based search (BM25) with semantic, meaning-based matching (sentence embeddings), so "AI Engineer" and "Machine Learning Engineer" are recognized as the same thing.
5. Every score ships with a recruiter-readable explanation, not just a number.

---

## Industry Scope

Right now FitHire can process other job fields, but it is currently **optimized for engineering and technical hiring** - the role-detection model, weight amplifiers, and scoring heuristics in `main.py` are tuned for engineering archetypes (ML, backend, frontend, data, DevOps, security, management).

That said, the foundation underneath isn't engineering-specific. The AI JD parser (`analyze_jd_with_ai` / `extract_jd_heuristic`) extracts **generic hiring concepts**, not engineering-only ones:

- `role_title`
- `must_have_skills`
- `years_min` / `years_max`
- `mandatory_certifications`
- `soft_skills`
- `education_requirement`
- `preferred_companies`
- `domain_keywords`
- `key_responsibilities`

Because these fields are domain-agnostic, FitHire can already parse and rank candidates for non-engineering JDs - it just won't yet have a specialized weight profile or role detector tuned for, say, a Sales or Legal hire the way it does for an ML Engineer. See **Future Scope → Multi-Domain Recruitment Support** below for where this is headed.

---

## Core Features

✔ AI-powered Job Description understanding (LLM mode with heuristic regex fallback when no API key is present)

✔ Dynamic, role-aware weight generation - no two job descriptions score candidates the same way

✔ Per-candidate adaptive weight redistribution - missing data is never penalized as zero

✔ Hybrid retrieval - BM25 keyword search + sentence-transformer semantic similarity + cross-encoder re-ranking

✔ Three-layer scoring model - **Technical Fit**, **Career Fit**, **Recruiter Fit**

✔ Career progression analysis - tenure stability, job-hopping detection, promotion trajectory, leadership and architecture-ownership signals

✔ Education scoring against JD requirements (penalizes under-qualification, rewards exceeding it)

✔ Built-in fairness filter - strips gender, age, religion, nationality, ethnicity, disability, and other protected attributes before any scoring happens

✔ Universal candidate normalization - accepts wildly inconsistent field names (`skills` / `technical_skills` / `tech_stack` …) and folds them into one schema

✔ Explainable, recruiter-grade reasoning generated per candidate

✔ CSV export in the challenge's required output format

---

## Architecture

FitHire consists of two interconnected layers: a core FastAPI backend processing infrastructure (`main.py`) that handles parsing, modeling, and deep evaluation, paired with a lightweight, standalone execution CLI engine (`rank_cli.py`) optimized for headless, high-throughput offline candidate evaluations and dataset exports.

```text
               ┌──────────────────────────────────────────────┐
               │         FitHire Orchestration Engine          │
               └──────────────────────┬───────────────────────┘
                                       │
              ┌────────────────────────┴───────────────────────┐
              ▼                                                 ▼
      [ Web Frontend UI ]                               [ Bulk Evaluation ]
     (FastAPI Local Server)                             (backend/rank_cli.py)
              │                                                 │
              └────────────────────────┬───────────────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │   1. Identity Anonymizer     │ ──► Strips Protected Attributes
                        └──────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │  2. Contextual JD Parser     │ ──► Generates Role Weight Profile
                        └──────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │ 3. Candidate Normalizer      │ ──► Resolves Profile Synonyms
                        └──────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │ 4. Adaptive Weight Allocator │ ──► Redistributes Missing Fields
                        └──────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │  5. Hybrid Scoring Engine    │
                        │   - BM25 Keyword Search      │
                        │   - Bi-Encoder Embedding     │
                        │   - Cross-Encoder Ranker     │ ──► Wall-Clock Budget Guard
                        └──────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │   6. Multi-Layer Blending    │ ──► Tech + Career + Recruiter Fit
                        └──────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │  7. Explainable Generation   │ ──► Recruiter-Grade Insight Strings
                        └──────────────┬──────────────┘
                                        ▼
                        ┌─────────────────────────────┐
                        │ 8. Output Dataset Export     │ ──► Valid .csv
                        └─────────────────────────────┘
```

### main.py vs rank_cli.py

| | `main.py` | `rank_cli.py` |
|---|---|---|
| Role | The chef - owns all the intelligence | The waiter - takes the order to the chef and brings results back |
| FastAPI server | ✅ | ❌ |
| JD parsing, candidate parsing, normalization | ✅ (owns it) | Calls into `main.py` |
| Scoring & ranking logic | ✅ (owns it) | Calls into `main.py` |
| Frontend integration | ✅ | ❌ |
| Bulk processing / CSV export for submission | Limited | ✅ |

### Scoring Pipeline in Detail

| Stage | What happens |
|---|---|
| Dependency loading | Optional libraries (PyPDF2, python-docx, rank-bm25, sentence-transformers) are detected at startup; the system still runs with reduced capability if any are missing |
| Fairness filter | Protected attributes (gender, age, religion, nationality, ethnicity, disability, sexual orientation, etc.) are stripped from every candidate before scoring begins |
| JD understanding | AI mode extracts role title, experience range, must-have/nice-to-have skills, certifications, responsibilities, preferred companies, and soft skills; heuristic mode (regex + keyword extraction) is the automatic fallback when no AI key is configured |
| Dynamic weight generation | Detects the role family from the JD (e.g. DevOps vs. People Manager vs. ML Engineer) and generates a custom weight profile instead of one-size-fits-all weights |
| Candidate normalization | Resolves dozens of field-name synonyms into one schema, auto-detects resume sections from unstructured text, and infers years of experience from a title when it isn't stated explicitly |
| Adaptive weight redistribution | Per candidate, any signal with no available data has its weight redistributed across the candidate's remaining signals instead of being scored as zero |
| Career progression analysis | Evaluates tenure stability, job-hopping patterns, promotion trajectory, leadership language ("led", "mentored", "owned"), and architecture-ownership signals |
| Education scoring | Compares candidate education against JD requirements, applying a penalty multiplier for under-qualification and a bonus for exceeding it |
| Three-layer scoring | Combines everything into Technical Fit, Career Fit, and Recruiter Fit sub-scores plus a final weighted score |
| Confidence + explanation | Computes a confidence score and generates a recruiter-readable reasoning string for every ranked candidate |

### Supported Formats

| Job Description | Candidate Pool |
|---|---|
| PDF | JSON |
| DOCX | JSONL |
| TXT | CSV |
| Paste directly | — |

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Single-page HTML/CSS/JS - resume-style landing page that scroll-reveals the upload workflow |
| Backend | FastAPI |
| Retrieval | BM25 (rank-bm25) for keyword search |
| Semantic search | sentence-transformers (all-MiniLM-L6-v2 bi-encoder) + cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) |
| JD/resume parsing | PyPDF2, python-docx, pandas/openpyxl |
| AI reasoning | LLM-based JD analysis with a heuristic regex fallback when offline |
| Export | CSV (challenge-required format) |

---

## Project Structure

```text
FitHire
│
├── backend/
│   ├── main.py              # Ranking engine + FastAPI API server
│   ├── rank_cli.py          # CLI wrapper for bulk/offline ranking + CSV export
│   └── __init__.py
│
├── frontend/
│   └── dist/
│       └── index.html       # Resume-style landing page + upload UI
│
├── data/
│   ├── job_description.txt  # Official released JD (bundled for a self-contained reproduce command)
│   ├── sample_jd.txt        # Demo JD, used only by the quick local smoke test
│   ├── sample_candidates.json
│   └── sample_candidates.csv
│
├── download_models.py       # One-time setup: pre-caches sentence-transformers models
├── requirements.txt
├── run.py                   # One-command launcher
└── README.md
```

---

## Quick Start

### 1. Create and activate an isolated virtual environment

Run everything for this project inside its own `.venv` — **do not** install into your global/system Python. Mixing this project's pinned dependency versions with other projects on the same interpreter is a real source of breakage (version conflicts, hard-to-diagnose import errors), not a hypothetical one.

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Your terminal prompt should now show `(.venv)` at the start of the line. Confirm before continuing.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Launch the web interface

To evaluate shortlists or test the upload workflow interactively:

```bash
python run.py
```

Then open **http://localhost:8000** in your browser.

---

## Running the Bulk CLI (Production Evaluation Sequence)

### Step 1: Pre-download and cache the models

The competition evaluation runs fully offline during ranking (`has_network_during_ranking: false`). Before running a timed production ranking pass, download and cache the model weights once:

```bash
python download_models.py
```

This caches `all-MiniLM-L6-v2` (bi-encoder) and `cross-encoder/ms-marco-MiniLM-L-6-v2` locally. From then on, `rank_cli.py` sets `HF_HUB_OFFLINE=1` automatically, so it reads only from that local cache and fails fast (instead of hanging) if a model isn't cached yet.

### Step 2: Run the full 100K-candidate reproduction

Replace `<PATH_TO_CANDIDATES>` below with the actual path to your copy of `candidates.jsonl`.

```bash
python -m backend.rank_cli \
  --candidates "C:\Users\Lenovo\OneDrive\Documents\Redrob AI\Data and AI challenge\FitHire\data\candidates.jsonl" \
  --jd "C:\Users\Lenovo\OneDrive\Documents\Redrob AI\Data and AI challenge\FitHire\data\job_description.txt" \
  --out team_sus.csv \
  --top 100
```

> **Windows Git Bash note:** if your path contains spaces or brackets (e.g. `[PUB] ...`), keep the whole `--candidates` value inside double quotes exactly as shown above.

**Time-budget guard:** the semantic scoring stage (bi-encoder + cross-encoder re-ranking) tracks its own elapsed time against the 5-minute wall-clock limit. If model inference is running too slowly to finish all 5,000 semantically-scored candidates in time, it automatically falls back to a fast heuristic score for whatever's left, so the run finishes within budget instead of risking a timeout disqualification.

### Step 3: Rename and validate

Rename the output to your registered participant ID:

```bash
cp team_sus.csv YOUR_PARTICIPANT_ID.csv
```

Run it through the organizer's validator (substitute the actual path to `validate_submission.py` from your hackathon bundle):

```bash
python "/validate_submission.py" YOUR_PARTICIPANT_ID.csv
```

A successful run prints: `Submission is valid.`

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/rank` | POST | Upload a JD (file or pasted text) and a candidate pool (file or JSON text); returns the full ranked shortlist with score breakdowns |
| `/api/export/csv` | POST / GET | Export a given set of ranked results to CSV |
| `/api/health` | GET | Health check - reports which optional capabilities (BM25, transformers) are currently active |

---

## Challenge Compatibility

Built specifically for **The Data & AI Challenge**.

- **Input:** a job description plus a candidate pool in any of the supported formats (including the challenge's `.jsonl` dataset).
- **Output:** `YOUR_PARTICIPANT_ID.csv` containing candidate ID, rank, overall score, confidence, AI-generated reasoning, matched skills, and the full Technical/Career/Recruiter Fit breakdown - ready to submit as-is.

---

## Future Scope

FitHire's recruiter explanation engine and confidence scoring are already built (see Core Features above) - the roadmap below is everything not yet in the codebase.

1. **Multi-Domain Recruitment Support** — Currently the system is optimized for engineering and technical roles. Future versions can include specialized ranking models for Law, Finance and Commerce, Healthcare, Marketing, Human Resources, and Sales and Consulting - each with its own customized scoring strategy and role-specific intelligence, building on the domain-agnostic JD parser described above.
2. **Learning from Recruiter Feedback** — Introduce a feedback loop where the system learns from recruiter actions - shortlisted candidates, rejected candidates, interview selections, final hires - so the ranking model continuously improves and adapts to a company's actual hiring preferences over time.
3. **Integration with Professional Platforms** — Integrate with LinkedIn, GitHub, LeetCode, and HackerRank to automatically enrich candidate profiles with live coding activity, projects, certifications, and professional achievements, rather than relying solely on recruiter-supplied data.
4. **AI Interview Assistant** — Extend the platform to automatically generate interview questions, create coding assessments, suggest case studies, and evaluate interview responses - turning FitHire from a candidate-ranking tool into an end-to-end hiring solution.
5. **Real-Time Market Intelligence** — Analyze current hiring trends to recommend emerging skills, salary ranges, market demand, and skill shortages - helping recruiters write better job descriptions and set realistic hiring strategies.
6. **Bias Detection and Fairness Monitoring** — The fairness engine already strips protected attributes (gender, age, religion, nationality, ethnicity, disability, etc.) before scoring. The next step is continuous monitoring on top of that - tracking outcomes for gender bias, age bias, educational bias, and institutional bias over time to support more inclusive hiring practices, not just point-in-time filtering.
7. **Video Resume and Portfolio Analysis** — Future versions can evaluate video resumes, portfolios, presentations, design work, and research publications - particularly useful for creative and non-technical roles where a text resume undersells the candidate.
8. **Support for Multiple Languages** — Enable resume parsing and ranking for resumes written in multiple languages, making the system viable for global recruitment rather than English-only pipelines.
9. **Predictive Hiring Analytics** — Use historical hiring data to predict candidate success probability, retention likelihood, promotion potential, and cultural fit.
10. **Automated Candidate Outreach** — Integrate with email and messaging systems to send interview invitations, schedule interviews, provide status updates, and automate recruiter communication end to end.
11. **Enterprise ATS Integration** — Integrate with existing recruitment platforms such as Workday, Greenhouse, and Lever, so FitHire can slot into a recruiter's existing tooling instead of running standalone.

Also planned: vector database support for million-scale candidate search, an LLM-powered recruiter copilot for follow-up questions like "why was this candidate selected over that one?", side-by-side candidate comparison dashboard improvements, and team collaboration / recruiter notes.

---

**Built for the Data & AI Challenge.**
Hiring deserves more than keyword matching. FitHire helps recruiters spend less time searching and more time hiring the right people.