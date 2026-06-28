# FitHire - Intelligent Candidate Ranking System

> Rank candidates the way a great recruiter would, not by matching keywords, but by actually understanding who fits the role.

## Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Start the server
python run.py

# 3. Open in browser
# http://localhost:8000
```

## What It Does

FitHire is a full-stack AI-powered recruiter tool that:

1. **Reads your Job Description** — PDF, DOCX, TXT, or paste directly
2. **Understands the role** — Claude AI extracts must-have skills, seniority, experience range, preferred backgrounds, and what actually matters
3. **Scores every candidate** across 6 dimensions:
   - **BM25 Semantic Match** — TF-IDF relevance of full profile against JD
   - **Skill Alignment** — weighted match of must-have vs nice-to-have skills
   - **Title Relevance** — is their background the right shape for this role?
   - **Experience Fit** — is their years of experience in the right range?
   - **Production Signals** — did they actually ship things, or is it all theory?
   - **Engagement** — are they responsive, active, and actually looking?
4. **Delivers a shortlist** with score breakdowns, matched skills, and human-readable reasoning

## Supported Input Formats

| Format | Job Description | Candidates |
|--------|----------------|------------|
| PDF    | ✅             | —          |
| DOCX   | ✅             | —          |
| TXT    | ✅             | —          |
| Paste  | ✅             | ✅ JSON   |
| JSONL  | —              | ✅         |
| JSON   | —              | ✅         |
| CSV    | —              | ✅         |
| Excel  | —              | ✅         |

## Scoring Formula

```
technical    = 0.35 × bm25 + 0.40 × skill_alignment + 0.25 × title_relevance
base_score   = 0.50 × technical + 0.25 × experience_fit + 0.15 × production_signals + 0.10 × assessments
engagement   = (0.70 + 0.30 × response_rate) × (0.95 + 0.05 × interview_rate) × open_to_work_bonus
final_score  = base_score × engagement + github_bonus + notice_bonus
```

## Architecture

```
recruiter-ai/
├── backend/
│   └── main.py          # FastAPI app — JD parsing, candidate normalization, ranking pipeline
├── frontend/
│   └── dist/
│       └── index.html   # Self-contained React UI
├── requirements.txt
├── run.py               # One-command startup
└── README.md
```

## API Endpoints

| Endpoint          | Method | Description                    |
|-------------------|--------|--------------------------------|
| `/api/rank`       | POST   | Upload JD + candidates, get ranked list |
| `/api/export/csv` | POST   | Export results as CSV          |
| `/api/health`     | GET    | Health check                   |

## Data Format

The system is compatible with the Redrob challenge JSONL schema but also handles:
- Flat CSV/Excel with columns: `name`, `current_title`, `years_experience`, `skills`, etc.
- Generic JSON with various field naming conventions

## Challenge Compatibility

This system is designed to work with the India Runs Data & AI Challenge dataset:
- Input: `candidates.jsonl` (100K+ candidates)
- Output: `ranked_candidates.csv` with `candidate_id`, `rank`, `score`, `reasoning`

To run against the challenge dataset via CLI:
```bash
python -m backend.rank_cli --candidates candidates.jsonl --jd job_description.txt --out submission.csv --top 100
```
