"""
FitHire — Intelligent Candidate Ranking Backend
====================================================
FastAPI backend that:
  1. Parses job descriptions (PDF, DOCX, TXT, plain text)
  2. Parses candidate data (JSON, JSONL, CSV, Excel)
  3. Runs a hybrid BM25 + multi-signal scoring pipeline
  4. Uses Claude AI for intelligent JD understanding + reasoning
  5. Returns ranked shortlist with explanations
"""

import os
import re
import csv
import json
import io
import time
import tempfile
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel



try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    import torch.nn.functional as F
    import math
    HAS_TRANSFORMERS = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
except ImportError:
    HAS_TRANSFORMERS = False

def parse_and_score_education(candidate_text: str, jd_analysis: Dict) -> float:
    text = candidate_text.lower()
    DEGREE_TIERS = {"phd": 4, "doctorate": 4, "master": 3, "ms": 3, "msc": 3, "mba": 3, "mtech": 3, "bachelor": 2, "bs": 2, "bsc": 2, "btech": 2, "ba": 2, "diploma": 1, "associate": 1}
    cand_tier = 0
    for degree, tier in DEGREE_TIERS.items():
        if re.search(r'\b' + re.escape(degree) + r'\b', text):
            cand_tier = max(cand_tier, tier)
    jd_text = jd_analysis.get("experience_focus", "").lower() + " " + " ".join(jd_analysis.get("key_responsibilities", [])).lower()
    req_tier = 2
    if any(w in jd_text for w in ["phd", "doctorate"]): req_tier = 4
    elif any(w in jd_text for w in ["master", "ms", "msc"]): req_tier = 3
    if cand_tier == 0: return 0.8
    elif cand_tier < req_tier: return 0.5
    elif cand_tier > req_tier: return 1.15
    return 1.0
# ──────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="FitHire", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TODAY = date.today()

# ──────────────────────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────────────────────
class RankRequest(BaseModel):
    jd_text: str
    candidates_json: str
    top_n: int = 20
    use_ai: bool = True


# ──────────────────────────────────────────────────────────────
# File Parsers
# ──────────────────────────────────────────────────────────────
def parse_jd_file(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        if not HAS_PDF:
            raise HTTPException(400, "PyPDF2 not installed")
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    elif ext == "docx":
        if not HAS_DOCX:
            raise HTTPException(400, "python-docx not installed")
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        return file_bytes.decode("utf-8", errors="replace")


def parse_candidates_file(file_bytes: bytes, filename: str) -> List[Dict]:
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "jsonl":
        candidates = []
        for line in file_bytes.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except:
                    pass
        return candidates

    elif ext == "json":
        data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for key in ("candidates", "data", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]

    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_dict(orient="records")

    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df.to_dict(orient="records")

    else:
        text = file_bytes.decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except:
            pass
        candidates = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except:
                    pass
        return candidates


# ──────────────────────────────────────────────────────────────
# JD Analysis (Claude-powered or heuristic)
# ──────────────────────────────────────────────────────────────
async def analyze_jd_with_claude(jd_text: str) -> Dict:
    """Call Claude API to extract structured requirements from JD."""
    import httpx

    prompt = f"""You are an expert technical recruiter. Analyze this job description and extract structured information.

JOB DESCRIPTION:
{jd_text[:4000]}

Respond ONLY with a JSON object (no markdown, no explanation) with this exact structure:
{{
  "role_title": "inferred job title",
  "seniority": "junior|mid|senior|lead|principal|manager",
  "years_min": 0,
  "years_max": 20,
  "must_have_skills": ["skill1", "skill2"],
  "nice_to_have_skills": ["skill3"],
  "key_responsibilities": ["resp1", "resp2"],
  "industry_signals": ["production", "startup", "research"],
  "red_flag_backgrounds": ["sales", "marketing"],
  "preferred_titles": ["software engineer", "ml engineer"],
  "domain_keywords": ["keyword1", "keyword2"],
  "experience_focus": "description of what kind of experience matters most",
  "preferred_companies": ["product companies", "startups"],
  "location_preference": "city or remote preference"
}}"""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            data = resp.json()
            text = data["content"][0]["text"]
            text = re.sub(r"```json|```", "", text).strip()
            return json.loads(text)
    except Exception as e:
        print(f"Claude JD analysis failed: {e}")
        return extract_jd_heuristic(jd_text)


def extract_jd_heuristic(jd_text: str) -> Dict:
    """Heuristic extraction from JD text."""
    text_lower = jd_text.lower()
    tokens = re.findall(r"[a-z0-9\+\#\.]+", text_lower)

    years_min, years_max = 2, 10
    range_match = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)", text_lower)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        if 1 <= lo <= 30 and 1 <= hi <= 30:
            years_min, years_max = lo, hi
    else:
        single_years = [int(y) for y in re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text_lower) if 1 <= int(y) <= 30]
        if single_years:
            years_min = min(single_years)
            years_max = max(single_years) if len(single_years) > 1 else years_min + 3

    seniority = "mid"
    if any(w in text_lower for w in ["senior", "sr.", "lead", "principal"]):
        seniority = "senior"
    elif any(w in text_lower for w in ["junior", "jr.", "entry", "fresher"]):
        seniority = "junior"
    elif any(w in text_lower for w in ["manager", "head of", "director"]):
        seniority = "manager"

    tech_pool = {
        "python", "java", "javascript", "typescript", "go", "rust", "scala", "kotlin",
        "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring",
        "pytorch", "tensorflow", "keras", "sklearn", "xgboost", "lightgbm",
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
        "sql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "spark", "airflow", "flink", "dbt",
        "llm", "rag", "embeddings", "faiss", "pinecone", "milvus", "weaviate", "qdrant",
        "transformers", "bert", "gpt", "huggingface",
        "mlflow", "wandb", "kubeflow", "sagemaker",
        "git", "nlp", "deep learning", "machine learning", "ai",
    }
    found_skills = [t for t in tokens if t in tech_pool]

    return {
        "role_title": "Software Engineer",
        "seniority": seniority,
        "years_min": years_min,
        "years_max": years_max,
        "must_have_skills": list(dict.fromkeys(found_skills))[:10],
        "nice_to_have_skills": list(dict.fromkeys(found_skills))[10:20],
        "key_responsibilities": [],
        "industry_signals": ["production", "deployed", "shipped"],
        "red_flag_backgrounds": ["sales", "marketing", "hr"],
        "preferred_titles": [],
        "domain_keywords": list(dict.fromkeys(t for t in tokens if len(t) > 2 and t.isalpha()))[:40],
        "experience_focus": "hands-on engineering experience",
        "preferred_companies": [],
        "location_preference": "",
    }


# ──────────────────────────────────────────────────────────────
# Candidate Normalization — handles flat CSV and nested JSON
# ──────────────────────────────────────────────────────────────
def normalize_candidate(raw: Dict) -> Dict:
    """Normalize various candidate formats into a unified schema."""

    # Already in structured format (nested)
    if "profile" in raw and isinstance(raw.get("profile"), dict):
        c = raw.copy()
        # ensure candidate_id exists
        if "candidate_id" not in c:
            c["candidate_id"] = str(id(raw))
        return c

    # Flat format (CSV / simple JSON)
    def get(*keys, default=""):
        for k in keys:
            for variant in [k, k.lower(), k.upper(), k.title()]:
                v = raw.get(variant)
                if v is not None and str(v).strip() not in ("", "nan", "None"):
                    return str(v).strip()
        return default

    candidate_id = get("candidate_id", "id", "ID", "CandidateID", default=f"CAND_{abs(id(raw))}")
    name = get("name", "Name", "full_name", "FullName", "anonymized_name", default="Unknown")
    headline = get("headline", "Headline", "title", "Title", "job_title", default="")
    summary = get("summary", "Summary", "about", "bio", "Bio", "description", default="")
    current_title = get("current_title", "CurrentTitle", "role", "Role", "position", "Position", default=headline)
    current_company = get("current_company", "CurrentCompany", "company", "Company", "employer", default="")
    location = get("location", "Location", "city", "City", default="")

    years_exp = 0.0
    for k in ["years_of_experience", "years_experience", "experience", "YearsExperience", "total_experience", "exp"]:
        v = raw.get(k)
        if v is not None:
            try:
                years_exp = float(str(v).replace("years", "").replace("yrs", "").strip())
                break
            except:
                pass

    # Skills — handle comma/semicolon string or list
    skills_raw = None
    for k in ["skills", "Skills", "skill_set", "skillset", "technologies"]:
        if k in raw:
            skills_raw = raw[k]
            break

    skills = []
    if isinstance(skills_raw, list):
        for s in skills_raw:
            if isinstance(s, dict):
                skills.append(s)
            else:
                skills.append({"name": str(s).strip(), "proficiency": "intermediate"})
    elif isinstance(skills_raw, str) and skills_raw:
        for s in re.split(r"[,;|]", skills_raw):
            s = s.strip()
            if s:
                skills.append({"name": s, "proficiency": "intermediate"})

    # Work experience from flat fields
    career = []
    if "career_history" in raw and isinstance(raw["career_history"], list):
        career = raw["career_history"]
    elif "work_experience" in raw and isinstance(raw["work_experience"], list):
        career = raw["work_experience"]
    elif current_title or current_company or summary:
        # Build synthetic career entry from flat fields so scoring has content
        career = [{
            "title": current_title,
            "company": current_company,
            "description": summary,
            "duration_months": int(years_exp * 12),
        }]

    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": name,
            "headline": headline,
            "summary": summary,
            "location": location,
            "country": get("country", "Country", default=""),
            "years_of_experience": years_exp,
            "current_title": current_title,
            "current_company": current_company,
            "current_company_size": get("current_company_size", "company_size", default=""),
            "current_industry": get("current_industry", "industry", "Industry", default=""),
        },
        "career_history": career,
        "education": raw.get("education", []) if isinstance(raw.get("education"), list) else [],
        "skills": skills,
        "certifications": raw.get("certifications", []) if isinstance(raw.get("certifications"), list) else [],
        "redrob_signals": raw.get("redrob_signals", {
            "last_active_date": "2024-06-01",
            "recruiter_response_rate": 0.7,
            "interview_completion_rate": 0.7,
            "open_to_work_flag": True,
            "github_activity_score": -1,
            "notice_period_days": 60,
            "profile_completeness_score": 70,
        }),
        "_raw_text": summary,  # keep raw text for scoring
    }


# ──────────────────────────────────────────────────────────────
# Scoring Helpers
# ──────────────────────────────────────────────────────────────

def build_candidate_text(c: Dict) -> str:
    """Build a full-text blob from all candidate fields for BM25 and matching."""
    profile = c.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
        profile.get("current_company", ""),
        c.get("_raw_text", ""),
    ]
    for role in c.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
        parts.append(role.get("company", ""))
    for skill in c.get("skills", []):
        name = skill.get("name", "") if isinstance(skill, dict) else str(skill)
        parts.append(name)
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", "") if isinstance(cert, dict) else str(cert))
    return " ".join(parts)


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def days_since(date_str: str) -> int:
    try:
        return (TODAY - datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()).days
    except:
        return 9999


def exp_score(years: float, min_years: int, max_years: int) -> float:
    """Score based on experience range from JD — penalize extremes."""
    if min_years <= years <= max_years:
        return 1.0
    elif years < min_years:
        gap = min_years - years
        return max(0.2, 1.0 - gap * 0.15)
    else:
        gap = years - max_years
        return max(0.55, 1.0 - gap * 0.05)


def title_relevance(title: str, preferred_titles: List[str], red_flag_bgs: List[str]) -> float:
    t = title.lower()
    for rf in red_flag_bgs:
        if rf.lower() in t:
            return 0.05
    for pt in preferred_titles:
        if pt.lower() in t:
            return 1.0
    if any(w in t for w in ["engineer", "developer", "scientist", "architect", "lead", "head"]):
        return 0.85
    if any(w in t for w in ["analyst", "specialist", "consultant"]):
        return 0.55
    if any(w in t for w in ["manager", "coordinator"]):
        return 0.45
    return 0.35


def skill_alignment_score(
    candidate_skills: List,
    must_have: List[str],
    nice_to_have: List[str],
    full_text: str,
) -> Tuple[float, List[str]]:
    """
    Accurate skill matching:
    - Check explicit skills list (exact match + substring)
    - Check full text blob for mentions (weaker signal)
    Returns (normalized_score 0-1, list_of_matched_skill_names)
    """
    # Build skill name set (lowercase)
    skill_names = set()
    for s in candidate_skills:
        if isinstance(s, dict):
            skill_names.add(s.get("name", "").lower().strip())
        else:
            skill_names.add(str(s).lower().strip())

    text_lower = full_text.lower()

    matched = []
    score = 0.0

    for skill in must_have:
        skill_l = skill.lower().strip()
        if not skill_l:
            continue
        # Exact or substring in skills list
        if any(skill_l == sn or skill_l in sn or sn in skill_l for sn in skill_names):
            score += 3.0
            matched.append(skill)
        elif re.search(r'\b' + re.escape(skill_l) + r'\b', text_lower):
            # Found in full text (summary, career history)
            score += 1.2
            matched.append(f"~{skill}")

    for skill in nice_to_have:
        skill_l = skill.lower().strip()
        if not skill_l:
            continue
        if any(skill_l == sn or skill_l in sn or sn in skill_l for sn in skill_names):
            score += 1.0
            matched.append(skill)
        elif re.search(r'\b' + re.escape(skill_l) + r'\b', text_lower):
            score += 0.4

    max_possible = max(1.0, len(must_have) * 3.0 + len(nice_to_have) * 1.0)
    return min(1.0, score / max_possible), matched[:12]


def production_signal_score(text: str) -> float:
    """
    Detect production/shipped evidence vs academic/research-only.
    Returns 0.0–1.0.
    """
    text_l = text.lower()

    STRONG_POS = [
        "shipped", "production", "deployed", "deployment", "scaling", "scaled",
        "a/b test", "a/b testing", "launched", "released", "migrated",
        "real-time", "realtime", "1m ", "million users", "billion",
        "latency", "monitoring", "serving", "inference at scale",
        "reduced", "improved by", "optimized", "processing \\d+",
    ]
    WEAK_POS = [
        "built", "implemented", "integrated", "automated",
        "pipeline", "feature", "model in production",
    ]
    NEG = [
        "paper", "research only", "no production", "theoretical",
        "hypothesis", "academic", "arxiv", "published paper",
    ]

    pos_strong = sum(1 for p in STRONG_POS if re.search(p, text_l))
    pos_weak = sum(1 for p in WEAK_POS if p in text_l)
    neg = sum(1 for n in NEG if n in text_l)

    raw = pos_strong * 2.0 + pos_weak * 0.5 - neg * 2.0
    # Normalize: 0 raw → 0.35 (neutral), 6 raw → 1.0
    score = (raw + 3.5) / 9.5
    return max(0.0, min(1.0, score))


def company_type_score(company: str, preferred_companies: List[str], jd_text: str) -> float:
    """
    Score company type fit.
    Detects product vs service company, startup vs enterprise.
    """
    c = company.lower()
    jd_l = jd_text.lower()

    # Known service companies (IT consulting/outsourcing)
    service_cos = {"tcs", "infosys", "wipro", "accenture", "cognizant", "hcl", "capgemini",
                   "tech mahindra", "mphasis", "ltimindtree", "hexaware"}
    known_product = {"google", "amazon", "microsoft", "meta", "apple", "netflix", "uber",
                     "airbnb", "stripe", "razorpay", "swiggy", "zomato", "meesho", "zepto",
                     "cred", "phonepe", "paytm", "freshworks", "zoho", "browserstack"}

    prefers_product = "product company" in jd_l or "not service" in jd_l or "not consulting" in jd_l
    prefers_startup = "startup" in jd_l or "fast-growing" in jd_l or "early stage" in jd_l

    if any(sc in c for sc in service_cos):
        return 0.3 if prefers_product else 0.6
    if any(pc in c for pc in known_product):
        return 1.0

    # Unknown company — neutral
    return 0.72


def engagement_score(signals: Dict) -> float:
    rrr = float(signals.get("recruiter_response_rate", 0.7))
    icr = float(signals.get("interview_completion_rate", 0.7))
    open_flag = 1.04 if signals.get("open_to_work_flag", True) else 1.0
    return min(1.15, (0.65 + 0.35 * rrr) * (0.95 + 0.05 * icr) * open_flag)


def notice_period_bonus(signals: Dict) -> float:
    np_days = int(signals.get("notice_period_days", 60))
    if np_days <= 15:
        return 0.05
    elif np_days <= 30:
        return 0.03
    elif np_days <= 60:
        return 0.01
    return 0.0


def github_bonus(signals: Dict) -> float:
    gh = signals.get("github_activity_score", -1)
    if gh is not None and gh > 0:
        return min(0.06, gh / 100.0 * 0.06)
    return 0.0


# ──────────────────────────────────────────────────────────────
# Core Scoring Function
# ──────────────────────────────────────────────────────────────
def score_candidate(
    c: Dict,
    bm25_raw: float,
    bm25_max: float,
    jd_analysis: Dict,
    jd_text: str,
) -> Tuple[float, Dict]:
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {})
    skills = c.get("skills", [])
    full_text = build_candidate_text(c)

    # ── 1. BM25 relevance (normalized against the top score in this batch) ──
    bm25_s = 0.0
    if bm25_max > 0:
        bm25_s = min(1.0, bm25_raw / bm25_max)

    # ── 2. Skill alignment (most important signal) ──
    skill_s, matched_skills = skill_alignment_score(
        skills,
        jd_analysis.get("must_have_skills", []),
        jd_analysis.get("nice_to_have_skills", []),
        full_text,
    )

    # ── 3. Title relevance ──
    title_s = title_relevance(
        profile.get("current_title", ""),
        jd_analysis.get("preferred_titles", []),
        jd_analysis.get("red_flag_backgrounds", []),
    )

    # ── 4. Experience fit ──
    years = float(profile.get("years_of_experience", 0))
    exp_s = exp_score(years, jd_analysis.get("years_min", 2), jd_analysis.get("years_max", 10))

    # ── 5. Production signals ──
    ship_s = production_signal_score(full_text)

    # ── 6. Company type fit ──
    company_s = company_type_score(
        profile.get("current_company", ""),
        jd_analysis.get("preferred_companies", []),
        jd_text,
    )

    # ── 7. Engagement multiplier ──
    eng = engagement_score(signals)

    # ── 8. Bonuses ──
    gh_b = github_bonus(signals)
    notice_b = notice_period_bonus(signals)

    # ── 9. Assessment scores (if present) ──
    assessments = signals.get("skill_assessment_scores", {})
    assess_s = 0.0
    if assessments:
        assess_s = min(1.0, sum(assessments.values()) / (len(assessments) * 100))

    # ── [NEW] AI Semantic Multiplier ──
    semantic_multiplier = 1.0
    if HAS_TRANSFORMERS:
        full_text = build_candidate_text(c)
        jd_embedding = bi_encoder.encode(jd_text, convert_to_tensor=True)
        cand_embedding = bi_encoder.encode(full_text, convert_to_tensor=True)
        cos_sim = F.cosine_similarity(jd_embedding, cand_embedding, dim=0).item()
        cross_score = cross_encoder.predict([jd_text[:1500], full_text[:1000]])
        normalized_cross = 1 / (1 + math.exp(-cross_score))
        semantic_multiplier = (0.4 * cos_sim) + (0.6 * normalized_cross)

    # ── [NEW] Education Modifier ──
    edu_mod = parse_and_score_education(full_text, jd_analysis)

    # ── Final composite score ──
    technical = (0.30 * bm25_s + 0.45 * skill_s + 0.25 * title_s)
    base = (0.40 * technical + 0.20 * exp_s + 0.20 * ship_s + 0.12 * company_s + 0.08 * assess_s)

    # Blend base heuristic score with AI Semantic Multiplier
    if HAS_TRANSFORMERS:
        final = (base * 0.4) + (base * semantic_multiplier * 0.6)
    else:
        final = base

    # Apply engagement, bonuses, and education modifier
    final = min(1.0, (final * eng + gh_b + notice_b) * edu_mod)

    # ── Clamping ──
    if skill_s < 0.05 and bm25_s < 0.05 and semantic_multiplier < 0.3:
        final = min(final, 0.18)

    breakdown = {
        "bm25_match": round(bm25_s, 3),
        "skill_alignment": round(skill_s, 3),
        "title_relevance": round(title_s, 3),
        "experience_fit": round(exp_s, 3),
        "production_signals": round(ship_s, 3),
        "company_fit": round(company_s, 3),
        "engagement": round(eng, 3),
    }

    info = {
        "candidate_id": c["candidate_id"],
        "final_score": round(final, 4),
        "name": profile.get("anonymized_name", "Unknown"),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "years_exp": years,
        "location": profile.get("location", ""),
        "matched_skills": matched_skills,
        "breakdown": breakdown,
        "signals": {
            "open_to_work": signals.get("open_to_work_flag", True),
            "github_score": signals.get("github_activity_score", -1),
            "response_rate": signals.get("recruiter_response_rate", 0),
            "notice_days": signals.get("notice_period_days", 60),
            "completeness": signals.get("profile_completeness_score", 0),
            "last_active_days": days_since(signals.get("last_active_date", "2024-06-01")),
        },
    }
    return final, info


def build_reasoning(info: Dict, jd_analysis: Dict) -> str:
    parts = []
    bd = info["breakdown"]

    if bd["skill_alignment"] >= 0.75:
        parts.append("excellent skill match")
    elif bd["skill_alignment"] >= 0.50:
        parts.append("strong skill overlap")
    elif bd["skill_alignment"] >= 0.25:
        parts.append("partial skill match")

    if bd["production_signals"] >= 0.70:
        parts.append("proven production experience")
    elif bd["production_signals"] >= 0.50:
        parts.append("some production exposure")

    if bd["bm25_match"] >= 0.60:
        parts.append("highly relevant background")
    elif bd["bm25_match"] >= 0.35:
        parts.append("relevant background")

    if info["signals"].get("github_score", -1) > 60:
        parts.append("active open-source contributor")

    seniority = jd_analysis.get("seniority", "mid")
    exp_fit = bd["experience_fit"]
    if exp_fit >= 0.95:
        parts.append(f"ideal {seniority}-level experience ({info['years_exp']:.0f} yrs)")
    elif exp_fit >= 0.70:
        parts.append(f"{info['years_exp']:.0f} yrs experience")
    elif info["years_exp"] < jd_analysis.get("years_min", 2):
        parts.append(f"below required experience ({info['years_exp']:.0f} yrs)")
    elif info["years_exp"] > jd_analysis.get("years_max", 10):
        parts.append(f"over-experienced ({info['years_exp']:.0f} yrs)")

    matched = [m.lstrip("~") for m in info.get("matched_skills", [])[:4]]
    if matched:
        parts.append(f"skills: {', '.join(matched)}")

    if bd["company_fit"] < 0.4:
        parts.append("service company background")

    company = info["current_company"]
    title = info["current_title"]
    prefix = f"{title}{' at ' + company if company else ''}. " if title else ""

    if not parts:
        parts.append(f"profile reviewed for {jd_analysis.get('role_title', 'this role')}")

    return prefix + "; ".join(parts[:4]).capitalize() + "."


# ──────────────────────────────────────────────────────────────
# Main Ranking Pipeline
# ──────────────────────────────────────────────────────────────
async def run_ranking_pipeline(
    jd_text: str,
    raw_candidates: List[Dict],
    top_n: int = 20,
    use_ai: bool = True,
) -> Tuple[List[Dict], Dict]:

    # Step 1: Analyze JD
    if use_ai:
        jd_analysis = await analyze_jd_with_claude(jd_text)
    else:
        jd_analysis = extract_jd_heuristic(jd_text)

    print(f"[JD] Analyzed: {jd_analysis.get('role_title')} | must_have: {jd_analysis.get('must_have_skills', [])[:5]}")

    # JD tokens for BM25
    jd_tokens = tokenize_for_bm25(jd_text)
    for kw in jd_analysis.get("domain_keywords", []):
        jd_tokens.extend(tokenize_for_bm25(kw))
    for skill in jd_analysis.get("must_have_skills", []):
        jd_tokens.extend(tokenize_for_bm25(skill))

    # Step 2: Normalize candidates
    candidates = [normalize_candidate(c) for c in raw_candidates]
    print(f"[CAND] Normalized {len(candidates)} candidates")

    # Step 3: BM25 index
    if HAS_BM25:
        corpus = [tokenize_for_bm25(build_candidate_text(c)) for c in candidates]
        valid_pairs = [(i, doc) for i, doc in enumerate(corpus) if doc]

        if valid_pairs:
            valid_indices, valid_docs = zip(*valid_pairs)
            bm25_model = BM25Okapi(valid_docs)
            raw_scores = bm25_model.get_scores(jd_tokens)

            bm25_scores = [0.0] * len(candidates)
            for idx, score in zip(valid_indices, raw_scores):
                bm25_scores[idx] = float(score)

            bm25_max = max(bm25_scores) if any(s > 0 for s in bm25_scores) else 1.0
        else:
            bm25_scores = [0.0] * len(candidates)
            bm25_max = 1.0
    else:
        bm25_scores = [0.0] * len(candidates)
        bm25_max = 1.0

    print(f"[BM25] max={bm25_max:.2f}, scores sample={[round(s,2) for s in bm25_scores[:5]]}")

    # Step 4: Score all
    scored = []
    for i, c in enumerate(candidates):
        final, info = score_candidate(c, bm25_scores[i], bm25_max, jd_analysis, jd_text)
        scored.append((final, info))
        print(f"  {info['name']}: score={final:.3f} skill={info['breakdown']['skill_alignment']:.3f} bm25={info['breakdown']['bm25_match']:.3f}")

    # Step 5: Sort and take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    # Step 6: Build output
    results = []
    for rank, (score, info) in enumerate(top, 1):
        reasoning = build_reasoning(info, jd_analysis)
        results.append({
            "rank": rank,
            "candidate_id": info["candidate_id"],
            "score": round(score, 4),
            "name": info["name"],
            "current_title": info["current_title"],
            "current_company": info["current_company"],
            "years_experience": info["years_exp"],
            "location": info["location"],
            "skills_matched": info["matched_skills"],
            "reasoning": reasoning,
            "score_breakdown": info["breakdown"],
            "signals": info["signals"],
        })

    return results, jd_analysis


# ──────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────
@app.post("/api/rank")
async def rank_candidates(
    jd_file: Optional[UploadFile] = File(None),
    jd_text: Optional[str] = Form(None),
    candidates_file: Optional[UploadFile] = File(None),
    candidates_text: Optional[str] = Form(None),
    top_n: int = Form(20),
    use_ai: bool = Form(True),
):
    # Parse JD
    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        jd_content = parse_jd_file(jd_bytes, jd_file.filename)
    elif jd_text:
        jd_content = jd_text
    else:
        raise HTTPException(400, "Provide job description as file or text")

    if not jd_content.strip():
        raise HTTPException(400, "Job description is empty")

    # Parse candidates
    if candidates_file and candidates_file.filename:
        cand_bytes = await candidates_file.read()
        raw_candidates = parse_candidates_file(cand_bytes, candidates_file.filename)
    elif candidates_text:
        raw_candidates = json.loads(candidates_text)
        if not isinstance(raw_candidates, list):
            raw_candidates = [raw_candidates]
    else:
        raise HTTPException(400, "Provide candidates as file or JSON text")

    if not raw_candidates:
        raise HTTPException(400, "No candidates found in uploaded data")

    top_n = min(max(1, top_n), 500)

    results, jd_analysis = await run_ranking_pipeline(jd_content, raw_candidates, top_n, use_ai)

    return {
        "success": True,
        "total_candidates": len(raw_candidates),
        "shortlisted": len(results),
        "jd_analysis": jd_analysis,
        "results": results,
    }


@app.post("/api/export/csv")
async def export_csv(request: dict):
    results = request.get("results", [])
    return _make_csv_response(results)


@app.get("/api/export/csv")
async def export_csv_get(data: str):
    try:
        results = json.loads(data)
    except:
        raise HTTPException(400, "Invalid data")
    return _make_csv_response(results)


def _make_csv_response(results: List[Dict]):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "rank", "candidate_id", "name", "score", "current_title",
        "current_company", "years_experience", "location",
        "skills_matched", "reasoning",
        "bm25_match", "skill_alignment", "title_relevance",
        "experience_fit", "production_signals", "company_fit", "engagement",
    ])
    for r in results:
        bd = r.get("score_breakdown", {})
        writer.writerow([
            r.get("rank"), r.get("candidate_id"), r.get("name"),
            r.get("score"), r.get("current_title"), r.get("current_company"),
            r.get("years_experience"), r.get("location"),
            "; ".join(r.get("skills_matched", [])),
            r.get("reasoning"),
            bd.get("bm25_match"), bd.get("skill_alignment"),
            bd.get("title_relevance"), bd.get("experience_fit"),
            bd.get("production_signals"), bd.get("company_fit"), bd.get("engagement"),
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ranked_candidates.csv"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "bm25_available": HAS_BM25, "version": "2.0.0"}


# ──────────────────────────────────────────────────────────────
# Serve React frontend (built)
# ──────────────────────────────────────────────────────────────
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

static_dir = Path(__file__).parent.parent / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")