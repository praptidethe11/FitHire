<p align="center">
  <img src="image.png" alt="FitHire Banner" width="100%">
</p>

<h1 align="center">FitHire</h1>

<p align="center">
<b>Rank candidates the way a great recruiter would.</b><br>
Not by matching keywords, but by understanding people.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Data%20%26%20AI-Challenge-orange?style=for-the-badge)

</p>

---

# Why FitHire?

Recruiters don't reject good candidates because they lack talent.

They reject them because traditional ATS systems rely heavily on **keyword matching**.

Someone with years of Machine Learning experience might never be shortlisted simply because their resume says **"AI Engineer"** instead of **"Machine Learning Engineer."**

**FitHire changes that.**

Instead of asking:

> *"Do these words match?"*

it asks:

> **"Would this person actually succeed in this role?"**

---

# Features

✔ AI-powered Job Description Understanding

✔ Hybrid Candidate Ranking Engine

✔ Semantic Matching

✔ Skill Alignment

✔ Experience Analysis

✔ Explainable AI Recommendations

✔ CSV Export

✔ Supports PDF, DOCX, TXT, JSON, CSV & Excel

---

# How It Works

```text
                 📄 Job Description
                        │
                        ▼
            AI understands the role
      ─────────────────────────────────
      • Required Skills
      • Experience Range
      • Seniority
      • Preferred Background
      ─────────────────────────────────
                        │
                        ▼
             Candidate Normalization
                        │
                        ▼
          Hybrid AI Ranking Engine
         ┌───────────────────────────┐
         │ Semantic Matching         │
         │ Skill Alignment           │
         │ Title Relevance           │
         │ Experience Fit            │
         │ Production Signals        │
         │ Engagement Signals        │
         └───────────────────────────┘
                        │
                        ▼
         Explainable Candidate Scores
                        │
                        ▼
          🏆 Ranked Recruiter Shortlist
```

---

# Demo

> Replace the placeholder below with your screen recording.

```md
![FitHire Demo](docs/demo.gif)
```

The demo walks through:

- Uploading a Job Description
- Uploading Candidate Profiles
- AI-based Candidate Ranking
- Explainable Score Breakdown
- Exporting Results

---

# Scoring Pipeline

Every candidate is evaluated across multiple dimensions.

| Component | Purpose |
|------------|----------|
| Semantic Match | Understands context instead of keywords |
| Skill Alignment | Required vs Preferred Skills |
| Title Relevance | Career similarity |
| Experience Fit | Right experience level |
| Production Signals | Real-world impact |
| Engagement Signals | Availability & recruiter activity |

Final ranking combines these signals into a recruiter-friendly score.

---

# Supported Formats

| Job Description | Candidates |
|-----------------|------------|
| PDF | JSON |
| DOCX | JSONL |
| TXT | CSV |
| Paste | Excel |

---

# Tech Stack

| Layer | Technology |
|---------|------------|
| Frontend | React |
| Backend | FastAPI |
| AI | Claude AI |
| Ranking | BM25 + Hybrid Scoring |
| Parsing | PDF / DOCX / TXT |
| Export | CSV |

---

# Project Structure

```text
FitHire
│
├── backend/
│   ├── main.py
│   ├── ranking.py
│   ├── parser.py
│   └── ...
│
├── frontend/
│   └── dist/
│
├── requirements.txt
├── run.py
└── README.md
```

---

# Quick Start

```bash
# Clone the repository

git clone <repository>

# Install dependencies

pip install -r requirements.txt

# Start the server

python run.py
```

Open

```
http://localhost:8000
```

---

# API

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/api/rank` | POST | Rank candidates |
| `/api/export/csv` | POST | Export ranked results |
| `/api/health` | GET | Health Check |

---

# Challenge Compatibility

Designed for the **Data & AI Challenge**.

Input

```
candidates.jsonl
```

Output

```
ranked_candidates.csv
```

with

- Candidate ID
- Rank
- Score
- AI-generated Reasoning

---

# Future Improvements

- 🔹 Vector Database support for million-scale candidate search
- 🔹 Cross-Encoder re-ranking for stronger semantic understanding
- 🔹 LLM-powered recruiter copilot ("Why was this candidate selected?")
- 🔹 Candidate comparison dashboard
- 🔹 ATS integrations (Greenhouse, Lever, Workday)
- 🔹 Fairness & bias auditing
- 🔹 Team collaboration and recruiter notes

---

# Built for the Data & AI Challenge

**Hiring deserves more than keyword matching.**

FitHire helps recruiters spend less time searching and more time hiring the right people.
