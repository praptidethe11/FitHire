"""
FitHire — Adaptive Recruiter Intelligence Engine v3.0
====================================================
FastAPI backend that:
  1. Parses job descriptions (PDF, DOCX, TXT, plain text)
  2. Parses candidate data (JSON, JSONL, CSV, Excel)
  3. Runs an Adaptive Intelligence Scoring pipeline with:
     - Dynamic weight generation per JD
     - Per-candidate adaptive weight redistribution
     - Three-layer scoring (Technical / Career / Recruiter)
     - Explainable recruiter-grade reasoning
  4. Uses AI for intelligent JD understanding + reasoning
  5. Returns ranked shortlist with full breakdowns
"""

import os
import re
import csv
import json
import io
import time
import tempfile
import math
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
    HAS_TRANSFORMERS = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ──────────────────────────────────────────────────────────────
# Phase 10 — Fairness Module
# ──────────────────────────────────────────────────────────────
PROTECTED_ATTRIBUTES = [
    "gender", "age", "religion", "nationality", "ethnicity", "race",
    "marital_status", "disability", "sexual_orientation", "birth_date",
    "sex", "name", "anonymized_name"
]

def apply_fairness_filter(candidate_raw: Dict) -> Dict:
    """Remove protected attributes from candidate data before scoring."""
    filtered = candidate_raw.copy()
    DISPLAY_ONLY = {"name", "anonymized_name"}
    for attr in PROTECTED_ATTRIBUTES:
        if attr in filtered and attr not in DISPLAY_ONLY:
            del filtered[attr]
    if "profile" in filtered and isinstance(filtered["profile"], dict):
        profile = filtered["profile"].copy()
        for attr in PROTECTED_ATTRIBUTES:
            if attr in profile and attr not in DISPLAY_ONLY:
                del profile[attr]
        filtered["profile"] = profile
    return filtered

# ──────────────────────────────────────────────────────────────
# Phase 3 — Dynamic Weight Generator
# ──────────────────────────────────────────────────────────────
# Base signal catalogue — all weights must sum to 1.0
_BASE_WEIGHTS = {
    "skill_alignment":    0.27,
    "experience_fit":     0.16,
    "career_progression": 0.14,
    "semantic_match":     0.07,
    "production_signals": 0.06,
    "notice_period":      0.07,
    "company_fit":        0.04,
    "bm25_match":         0.04,
    "education":          0.03,
    "github_oss":         0.03,
    "title_relevance":    0.04,   # was computed but never weighted — now a real signal
    "certifications":     0.03,   # new first-class signal
}

# Role-type amplifier tables — these SHIFT relative importance, not hard-code weights
_ROLE_AMPLIFIERS = {
    "ml": {
        "skill_alignment": 1.30, "experience_fit": 1.20, "production_signals": 1.40,
        "semantic_match": 1.10, "career_progression": 1.10, "github_oss": 1.20,
        "certifications": 0.80,"title_relevance": 1.25,
    },
    "research": {
        "education": 2.50, "github_oss": 1.80, "semantic_match": 1.30,
        "production_signals": 0.60, "notice_period": 0.70, "certifications": 0.70,
    },
    "backend": {
        "skill_alignment": 1.25, "experience_fit": 1.15, "production_signals": 1.30,
        "career_progression": 1.10, "certifications": 1.10,"title_relevance": 1.15,
    },
    "frontend": {
        "skill_alignment": 1.30, "semantic_match": 1.10, "production_signals": 1.10,
        "title_relevance": 1.10,
    },
    "data": {
        "skill_alignment": 1.25, "experience_fit": 1.10, "production_signals": 1.20,
        "semantic_match": 1.15, "certifications": 1.30,
    },
    "devops": {
        "skill_alignment": 1.30, "production_signals": 1.50, "experience_fit": 1.10,
        "career_progression": 1.05, "certifications": 1.50,
    },
    "manager": {
        "career_progression": 1.60, "experience_fit": 1.20, "company_fit": 1.30,
        "skill_alignment": 0.80, "notice_period": 1.10, "title_relevance": 1.35,
    },
    "security": {
        "skill_alignment": 1.40, "experience_fit": 1.15, "production_signals": 1.20,
        "certifications": 2.00,
    },
    "generic": {},
}

def _detect_role_type(jd_analysis: Dict, jd_text: str) -> str:
    """Classify JD into a role archetype for amplifier selection."""
    text = jd_text.lower()
    title = jd_analysis.get("role_title", "").lower()
    must = " ".join(jd_analysis.get("must_have_skills", [])).lower()
    combined = f"{title} {must} {text[:1500]}"

    ml_kw  = ["machine learning", "deep learning", "neural", "pytorch", "tensorflow",
               "mlops", "model", "training", "inference", "llm", "nlp", "huggingface"]
    res_kw = ["research", "publication", "arxiv", "phd", "paper", "academia", "thesis"]
    be_kw  = ["backend", "api", "microservice", "spring", "django", "fastapi", "rest",
               "grpc", "kafka", "database", "postgres", "redis"]
    fe_kw  = ["frontend", "react", "angular", "vue", "css", "ui", "ux", "webpack",
               "javascript", "typescript"]
    da_kw  = ["data engineer", "spark", "airflow", "dbt", "warehouse", "pipeline",
               "etl", "bigquery", "snowflake", "analytics"]
    dv_kw  = ["devops", "kubernetes", "docker", "terraform", "ci/cd", "ansible",
               "infrastructure", "sre", "reliability"]
    mg_kw  = ["manager", "management", "team lead", "director", "vp", "head of",
               "people manager", "hiring"]
    sc_kw  = ["security", "penetration", "vulnerability", "soc", "siem", "compliance",
               "appsec", "devsecops"]

    scores = {
        "ml":       sum(1 for k in ml_kw  if k in combined),
        "research": sum(1 for k in res_kw if k in combined),
        "backend":  sum(1 for k in be_kw  if k in combined),
        "frontend": sum(1 for k in fe_kw  if k in combined),
        "data":     sum(1 for k in da_kw  if k in combined),
        "devops":   sum(1 for k in dv_kw  if k in combined),
        "manager":  sum(1 for k in mg_kw  if k in combined),
        "security": sum(1 for k in sc_kw  if k in combined),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "generic"

def generate_jd_weights(jd_analysis: Dict, jd_text: str) -> Dict[str, float]:
    """
    Phase 3: Generate dynamic weights from JD.
    Start from base weights, apply role-type amplifiers, renormalise to 1.0.
    """
    role_type = _detect_role_type(jd_analysis, jd_text)
    amplifiers = _ROLE_AMPLIFIERS.get(role_type, {})

    raw = {}
    for signal, base in _BASE_WEIGHTS.items():
        raw[signal] = base * amplifiers.get(signal, 1.0)

    total = sum(raw.values())
    normalised = {k: round(v / total, 6) for k, v in raw.items()}
    normalised["_role_type"] = role_type
    return normalised

def redistribute_weights(base_weights: Dict[str, float], available_signals: Dict[str, bool]) -> Dict[str, float]:
    """
    Phase 4: Adaptive weight redistribution.
    If a signal is missing, its weight is redistributed proportionally
    across the available signals. Total always sums to 1.0.
    Never penalise a candidate for missing data.
    """
    signal_keys = [k for k in base_weights if not k.startswith("_")]
    missing = [k for k in signal_keys if not available_signals.get(k, True)]
    present = [k for k in signal_keys if available_signals.get(k, True)]

    if not missing:
        return {k: base_weights[k] for k in signal_keys}

    lost_weight = sum(base_weights[k] for k in missing)
    present_total = sum(base_weights[k] for k in present)

    redistributed = {}
    for k in signal_keys:
        if k in missing:
            redistributed[k] = 0.0
        else:
            if present_total > 0:
                redistributed[k] = base_weights[k] + lost_weight * (base_weights[k] / present_total)
            else:
                redistributed[k] = 1.0 / len(present) if present else 0.0

    # Renormalise to exactly 1.0
    total = sum(redistributed.values())
    if total > 0:
        redistributed = {k: v / total for k, v in redistributed.items()}

    return redistributed

# ──────────────────────────────────────────────────────────────
# Phase 6 — Career Progression Engine
# ──────────────────────────────────────────────────────────────
def analyze_career_progression(career_history: List[Dict]) -> Tuple[float, Dict]:
    """
    Comprehensive career trajectory analysis.
    Returns (score_0_to_1, detail_dict) so callers get explainability data.
    Returns 0.0 when no history is available (weight redistributed by caller).
    """
    if not career_history:
        return 0.0, {"available": False}

    score = 0.50
    detail = {"available": True, "signals": []}

    # ── 1. Tenure stability ───────────────────────────────────
    total_months = sum(role.get("duration_months", 0) for role in career_history)
    avg_tenure = total_months / len(career_history) if career_history else 0
    if avg_tenure >= 36:
        score += 0.15
        detail["signals"].append("Stable long tenures (3+ yr avg)")
    elif avg_tenure >= 18:
        score += 0.06
        detail["signals"].append("Acceptable tenure (1.5–3 yr avg)")
    elif avg_tenure < 12 and len(career_history) >= 3:
        score -= 0.15
        detail["signals"].append("Job-hopping pattern detected")
    elif avg_tenure < 8 and len(career_history) >= 2:
        score -= 0.20
        detail["signals"].append("Severe job-hopping pattern")

    # ── 2. Seniority progression ──────────────────────────────
    seniority_levels = [
        "intern", "trainee", "junior", "associate", "mid", "specialist",
        "senior", "lead", "principal", "staff", "manager", "architect",
        "director", "vp", "head", "cto", "ceo"
    ]
    level_indices = []
    for role in career_history:
        title = role.get("title", "").lower()
        found = -1
        for i, lvl in enumerate(seniority_levels):
            if lvl in title:
                found = i
        level_indices.append(found)

    if len(level_indices) >= 2:
        valid = [l for l in level_indices if l != -1]
        if len(valid) >= 2:
            # Most-recent-first ordering assumed; valid[0] = most recent
            if valid[0] > valid[-1]:
                gap = valid[0] - valid[-1]
                score += min(0.22, 0.07 * gap)
                detail["signals"].append(f"Clear seniority progression (+{gap} levels)")
            elif valid[0] < valid[-1]:
                score -= 0.10
                detail["signals"].append("Career regression detected")
            else:
                detail["signals"].append("Lateral moves (same level)")

    # ── 3. Promotion speed signal ─────────────────────────────
    # If someone hit senior/lead within < 4 years total, that's a positive signal
    if total_months > 0 and level_indices:
        valid = [l for l in level_indices if l != -1]
        if valid and valid[0] >= 6:  # reached senior or above
            years_to_senior = total_months / 12
            if years_to_senior <= 4:
                score += 0.10
                detail["signals"].append(f"Fast promotion to senior level ({years_to_senior:.1f} yrs)")
            elif years_to_senior <= 7:
                score += 0.05
                detail["signals"].append("On-track promotion pace")

    # ── 4. Leadership, ownership & mentoring signals ──────────
    leadership_kw = [
        "managed", "led", "mentored", "architected", "founded", "owned",
        "responsible for team", "built team", "hired", "grew team",
        "technical lead", "tech lead", "engineering lead", "principal",
        "people manager", "line manager", "cross-functional",
    ]
    ownership_kw = [
        "took ownership", "drove", "spearheaded", "championed", "defined",
        "established", "designed the system", "designed architecture",
        "led the design", "owned the roadmap", "product owner",
    ]
    mentoring_kw = [
        "mentored", "coached", "onboarded", "upskilled", "trained junior",
        "knowledge transfer", "pair programming", "code review",
    ]
    architecture_kw = [
        "architected", "architecture", "system design", "designed the",
        "infrastructure", "platform design", "service mesh", "microservices",
        "distributed system", "scaled the", "re-architected",
    ]
    growth_kw = [
        "promoted", "took ownership", "expanded scope", "increased responsibility",
        "grew from", "transitioned to", "moved into",
    ]

    total_desc = " ".join(role.get("description", "") for role in career_history).lower()
    leadership_hits  = sum(1 for kw in leadership_kw  if kw in total_desc)
    ownership_hits   = sum(1 for kw in ownership_kw   if kw in total_desc)
    mentoring_hits   = sum(1 for kw in mentoring_kw   if kw in total_desc)
    architecture_hits = sum(1 for kw in architecture_kw if kw in total_desc)
    growth_hits      = sum(1 for kw in growth_kw      if kw in total_desc)

    score += min(0.15, leadership_hits * 0.05)
    score += min(0.08, ownership_hits * 0.04)
    score += min(0.06, mentoring_hits * 0.03)
    score += min(0.08, architecture_hits * 0.04)
    score += min(0.08, growth_hits * 0.04)

    if leadership_hits >= 2:
        detail["signals"].append(f"Leadership indicators ({leadership_hits} signals)")
    if ownership_hits >= 1:
        detail["signals"].append("Ownership/spearheaded signals")
    if mentoring_hits >= 1:
        detail["signals"].append("Mentoring/coaching signals")
    if architecture_hits >= 2:
        detail["signals"].append("Architecture/system design signals")
    if growth_hits >= 1:
        detail["signals"].append("Explicit growth/promotion mentions")

    # ── 5. Penalty: regression + no growth ───────────────────
    if not any(l != -1 for l in level_indices) and total_months < 24:
        score -= 0.10
        detail["signals"].append("Short tenure, no role titles found")

    final = round(max(0.0, min(1.0, score)), 4)
    detail["score"] = final
    return final, detail

# ──────────────────────────────────────────────────────────────
# Education Scoring
# ──────────────────────────────────────────────────────────────
def parse_and_score_education(candidate_text: str, jd_analysis: Dict) -> Tuple[float, bool]:
    """Returns (modifier_float, has_education_data)."""
    text = candidate_text.lower()
    DEGREE_TIERS = {
        "phd": 4, "doctorate": 4, "master": 3, "ms": 3, "msc": 3,
        "mba": 3, "mtech": 3, "bachelor": 2, "bs": 2, "bsc": 2,
        "btech": 2, "ba": 2, "diploma": 1, "associate": 1
    }
    cand_tier = 0
    for degree, tier in DEGREE_TIERS.items():
        if re.search(r'\b' + re.escape(degree) + r'\b', text):
            cand_tier = max(cand_tier, tier)

    jd_focus = jd_analysis.get("experience_focus", "").lower()
    jd_resps = " ".join(jd_analysis.get("key_responsibilities", [])).lower()
    jd_combined = jd_focus + " " + jd_resps
    req_tier = 2
    if any(w in jd_combined for w in ["phd", "doctorate"]): req_tier = 4
    elif any(w in jd_combined for w in ["master", "ms ", "msc"]): req_tier = 3

    has_data = cand_tier > 0
    if not has_data:
        return 0.95, False   # No education data → near-neutral modifier
    if cand_tier < req_tier:
        return 0.88, True
    elif cand_tier > req_tier:
        return 1.08, True
    return 1.0, True

# ──────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="FitHire", version="3.0.0")

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
    try:
        if ext == "pdf":
            text_pages = []
            
            # Fallback 1: PyPDF2
            if HAS_PDF:
                try:
                    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                    for p in reader.pages:
                        t = p.extract_text()
                        if t:
                            text_pages.append(t)
                except Exception as py_err:
                    print(f"[WARN] PyPDF2 extraction failed: {py_err}")

            # Fallback 2: PyMuPDF (fitz)
            if not "".join(text_pages).strip():
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(stream=file_bytes, filetype="pdf")
                    text_pages = []
                    for page in doc:
                        t = page.get_text()
                        if t:
                            text_pages.append(t)
                except Exception as fitz_err:
                    print(f"[WARN] PyMuPDF extraction failed: {fitz_err}")

            # Fallback 3: pdfplumber
            if not "".join(text_pages).strip():
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        text_pages = []
                        for page in pdf.pages:
                            t = page.extract_text()
                            if t:
                                text_pages.append(t)
                except Exception as plumber_err:
                    print(f"[WARN] pdfplumber extraction failed: {plumber_err}")

            # Fallback 4: OCR (pdf2image + pytesseract)
            if not "".join(text_pages).strip() and HAS_OCR:
                try:
                    from pdf2image import convert_from_bytes
                    images = convert_from_bytes(file_bytes)
                    text_pages = []
                    for img in images:
                        t = pytesseract.image_to_string(img)
                        if t:
                            text_pages.append(t)
                except Exception as ocr_err:
                    print(f"[WARN] PDF OCR extraction failed: {ocr_err}")

            final_text = "\n".join(text_pages)
            if not final_text.strip():
                raise HTTPException(400, "Could not extract any text from the PDF. It may be secured, corrupted, or scanned without OCR tools installed.")
            return final_text
        elif ext == "docx":
            if not HAS_DOCX:
                raise HTTPException(400, "python-docx library is not installed on the server.")
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        elif ext in ("jpg", "jpeg", "png"):
            if not HAS_OCR:
                raise HTTPException(400, "OCR/Tesseract dependencies are not installed on the server.")
            image = Image.open(io.BytesIO(file_bytes))
            return pytesseract.image_to_string(image)
        elif ext in ("txt", "md"):
            return file_bytes.decode("utf-8-sig", errors="replace")
        else:
            return file_bytes.decode("utf-8-sig", errors="replace")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Error parsing job description file: {str(e)}")

def parse_candidates_file(file_bytes: bytes, filename: str) -> List[Dict]:
    ext = filename.lower().rsplit(".", 1)[-1]
    try:
        if ext == "jsonl":
            candidates = []
            for line in file_bytes.decode("utf-8-sig", errors="replace").splitlines():
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except:
                        pass
            return candidates
        elif ext == "json":
            data = json.loads(file_bytes.decode("utf-8-sig", errors="replace"))
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
            text = file_bytes.decode("utf-8-sig", errors="replace")
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
    except Exception as e:
        raise HTTPException(400, f"Error parsing candidate pool file: {str(e)}")

# ──────────────────────────────────────────────────────────────
# Phase 1 — Job Understanding (AI + Heuristic)
# ──────────────────────────────────────────────────────────────
async def analyze_jd_with_ai(jd_text: str) -> Dict:
    """Extract structured requirements from JD using AI or heuristic fallback."""

    api_key = os.getenv("OPENAI_API_KEY")

    # No API configured → immediately use heuristic parser
    if not api_key:
        result = extract_jd_heuristic(jd_text)
        result["_using_fallback"] = True
        return result

    try:
        from openai import OpenAI   # lazy import — only when API key present
        client = OpenAI(api_key=api_key)

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
  "mandatory_certifications": [],
  "preferred_certifications": [],
  "key_responsibilities": ["resp1", "resp2"],
  "industry_signals": ["production", "startup", "research"],
  "red_flag_backgrounds": ["sales", "marketing"],
  "preferred_titles": ["software engineer", "ml engineer"],
  "domain_keywords": ["keyword1", "keyword2"],
  "experience_focus": "description of what kind of experience matters most",
  "preferred_companies": ["product companies", "startups"],
  "location_preference": "city or remote preference",
  "soft_skills": ["communication", "leadership"],
  "leadership_expected": false,
  "notice_period_preference_days": 30,
  "education_requirement": "bachelor|master|phd|any"
}}"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        if response and response.choices:
            result = json.loads(response.choices[0].message.content)
            result["_using_fallback"] = False
            return result
    except Exception as e:
        print(f"AI JD analysis failed: {e}")

    result = extract_jd_heuristic(jd_text)
    result["_using_fallback"] = True
    return result

# Also expose under legacy name used in tests / CLI
async def analyze_jd_with_claude(jd_text: str) -> Dict:
    return await analyze_jd_with_ai(jd_text)

def extract_jd_heuristic(jd_text: str) -> Dict:
    """Heuristic extraction from JD text — Phase 1 fallback."""
    text_lower = jd_text.lower()
    tokens = re.findall(r"[a-z0-9\+\#\.]+", text_lower)

    # ── Role Title ────────────────────────────────────────────
    role_title = "Software Engineer"
    title_match = re.search(r"(?:job title|role|position)\s*[:]\s*(.*)", jd_text, re.IGNORECASE)
    if title_match:
        role_title = title_match.group(1).split("\n")[0].strip()
    else:
        lines = [l.strip() for l in jd_text.split("\n") if l.strip()]
        if lines:
            first_line = lines[0]
            if len(first_line) < 100:
                cleaned = re.sub(r"^(?:jd|job description|role|position)\s*[-—–:]*\s*", "", first_line, flags=re.IGNORECASE)
                if cleaned.strip():
                    role_title = cleaned.strip()

    # ── Experience range ──────────────────────────────────────
    years_min, years_max = 2, 10
    range_match = re.search(r"(\d+)\s*[-–—]\s*(\d+)\s*(?:years?|yrs?)", text_lower)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        if 1 <= lo <= 30 and 1 <= hi <= 30:
            years_min, years_max = lo, hi
    else:
        single_years = [int(y) for y in re.findall(r"(\d+)\+?\s*(?:years?|yrs?)", text_lower) if 1 <= int(y) <= 30]
        if single_years:
            years_min = min(single_years)
            years_max = max(single_years) if len(single_years) > 1 else years_min + 3

    # ── Seniority ─────────────────────────────────────────────
    seniority = "mid"
    if any(w in text_lower for w in ["senior", "sr.", "lead", "principal", "staff"]):
        seniority = "senior"
    elif any(w in text_lower for w in ["junior", "jr.", "entry", "entry-level", "fresher", "graduate", "intern"]):
        seniority = "junior"
    elif any(w in text_lower for w in ["manager", "head of", "director", "vp"]):
        seniority = "manager"

    # Adjust years_min for freshers
    if seniority == "junior" and any(w in text_lower for w in ["fresher", "intern", "0 years", "0-"]):
        years_min = 0

    # ── Leadership expectation ────────────────────────────────
    leadership_kw = ["lead a team", "manage engineers", "mentor", "people manager",
                     "team lead", "engineering manager", "tech lead"]
    leadership_expected = any(kw in text_lower for kw in leadership_kw)

    # ── Notice period preference ──────────────────────────────
    np_match = re.search(r"(\d+)\s*(?:day|week|month)s?\s*notice", text_lower)
    notice_pref = 30
    if np_match:
        val = int(np_match.group(1))
        if "week" in text_lower[np_match.start():np_match.end()+5]:
            val *= 7
        elif "month" in text_lower[np_match.start():np_match.end()+6]:
            val *= 30
        notice_pref = val

    # ── Education requirement ─────────────────────────────────
    edu_req = "bachelor"
    if any(w in text_lower for w in ["phd", "doctorate", "ph.d"]): edu_req = "phd"
    elif any(w in text_lower for w in ["master", "ms ", "m.s.", "msc", "m.tech", "mtech"]): edu_req = "master"

    # ── Skills ───────────────────────────────────────────────
    tech_pool = {
        "python", "java", "javascript", "typescript", "go", "rust", "scala", "kotlin", "c++", "c#", "c", "ruby", "php",
        "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring", "html", "css",
        "pytorch", "tensorflow", "keras", "sklearn", "scikit-learn", "xgboost", "lightgbm", "numpy", "pandas", "scipy",
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd",
        "sql", "postgres", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
        "kafka", "spark", "airflow", "flink", "dbt", "hadoop",
        "llm", "rag", "embeddings", "faiss", "pinecone", "milvus", "weaviate", "qdrant",
        "transformers", "bert", "gpt", "huggingface", "langchain", "llama",
        "mlflow", "wandb", "kubeflow", "sagemaker", "mlops",
        "git", "nlp", "deep learning", "machine learning", "ai", "computer vision", "data science",
    }
    found_skills = []
    for tech in tech_pool:
        if tech in ("c++", "c#"):
            pattern = r'\b' + re.escape(tech)
        elif tech.endswith('s') or tech in ("go", "r", "c"):
            pattern = r'\b' + re.escape(tech) + r'\b'
        else:
            pattern = r'\b' + re.escape(tech) + r's?\b'
        
        if re.search(pattern, text_lower):
            found_skills.append(tech)

    return {
        "role_title": role_title,
        "seniority": seniority,
        "years_min": years_min,
        "years_max": years_max,
        "must_have_skills": list(dict.fromkeys(found_skills))[:10],
        "nice_to_have_skills": list(dict.fromkeys(found_skills))[10:20],
        "mandatory_certifications": [],
        "preferred_certifications": [],
        "key_responsibilities": [],
        "industry_signals": ["production", "deployed", "shipped"],
        "red_flag_backgrounds": ["sales", "marketing", "hr"],
        "preferred_titles": [],
        "domain_keywords": list(dict.fromkeys(t for t in tokens if len(t) > 2 and t.isalpha()))[:40],
        "experience_focus": "hands-on engineering experience",
        "preferred_companies": [],
        "location_preference": "",
        "soft_skills": [],
        "leadership_expected": leadership_expected,
        "notice_period_preference_days": notice_pref,
        "education_requirement": edu_req,
    }

# ──────────────────────────────────────────────────────────────
# Phase 2 — Candidate Normalization
# ──────────────────────────────────────────────────────────────
def normalize_candidate(raw: Dict) -> Dict:
    """Normalize various candidate formats into unified schema."""
    if "profile" in raw and isinstance(raw.get("profile"), dict):
        c = raw.copy()
        if "candidate_id" not in c:
            c["candidate_id"] = str(id(raw))
        return c

    def get(*keys, default=""):
        for k in keys:
            for variant in [k, k.lower(), k.upper(), k.title()]:
                v = raw.get(variant)
                if v is not None and str(v).strip() not in ("", "nan", "None"):
                    return str(v).strip()
        return default

    candidate_id = get("candidate_id", "id", "ID", "CandidateID", "candidateId", default=f"CAND_{abs(id(raw))}")
    name = get("name", "Name", "full_name", "FullName", "anonymized_name", "candidate_name", "candidateName", "full name", "fullName", default="Unknown")
    headline = get("headline", "Headline", "title", "Title", "job_title", default="")
    summary = get("summary", "Summary", "about", "bio", "Bio", "description", default="")
    current_title = get("current_title", "CurrentTitle", "role", "Role", "position", "Position", default=headline)
    current_company = get("current_company", "CurrentCompany", "company", "Company", "employer", default="")
    location = get("location", "Location", "city", "City", default="")

    years_exp = 0.0
    for k in ["years_of_experience", "years_experience", "experience", "YearsExperience", "total_experience", "exp", "total_exp"]:
        v = raw.get(k)
        if v is not None:
            val_str = str(v).strip().lower()
            if val_str in ("fresher", "none", "nan", "null", "0"):
                years_exp = 0.0
                break
            try:
                # Clean up any trailing text like "years", "yrs", etc.
                cleaned_val = re.sub(r'(?i)\s*(?:years?|yrs?|yr)\b.*', '', val_str).strip()
                years_exp = float(cleaned_val)
                break
            except:
                pass

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

    career = []
    if "career_history" in raw and isinstance(raw["career_history"], list):
        career = raw["career_history"]
    elif "work_experience" in raw and isinstance(raw["work_experience"], list):
        career = raw["work_experience"]
    elif current_title or current_company or summary:
        career = [{
            "title": current_title,
            "company": current_company,
            "description": summary,
            "duration_months": int(years_exp * 12),
        }]

    # Normalize education to list of dicts
    edu_raw = raw.get("education", raw.get("Education", []))
    education = []
    if isinstance(edu_raw, list):
        for e in edu_raw:
            if isinstance(e, dict):
                education.append(e)
            else:
                education.append({"degree": str(e).strip()})
    elif isinstance(edu_raw, str) and edu_raw.strip():
        education.append({"degree": edu_raw.strip()})

    # Extract additional Phase 2 signals
    projects = raw.get("projects", raw.get("Projects", []))
    publications = raw.get("publications", raw.get("Publications", []))
    awards = raw.get("awards", raw.get("Awards", []))
    open_source = raw.get("open_source", raw.get("github_url", raw.get("github", "")))

    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": name,
            "headline": headline,
            "summary": summary,
            "location": location,
            "current_title": current_title,
            "current_company": current_company,
            "years_of_experience": years_exp,
            "current_company_size": get("current_company_size", "company_size", default=""),
            "current_industry": get("current_industry", "industry", "Industry", default=""),
        },
        "career_history": career,
        "education": education,
        "skills": skills,
        "certifications": raw.get("certifications", []) if isinstance(raw.get("certifications"), list) else [],
        "projects": projects if isinstance(projects, list) else [],
        "publications": publications if isinstance(publications, list) else [],
        "awards": awards if isinstance(awards, list) else [],
        "open_source": str(open_source) if open_source else "",
        "redrob_signals": raw.get("redrob_signals", {
            "last_active_date": "2024-06-01",
            "recruiter_response_rate": 0.7,
            "interview_completion_rate": 0.7,
            "open_to_work_flag": True,
            "github_activity_score": -1,
            "notice_period_days": 60,
            "profile_completeness_score": 70,
        }),
        "_raw_text": summary,
    }

# ──────────────────────────────────────────────────────────────
# Scoring Helpers
# ──────────────────────────────────────────────────────────────
def build_candidate_text(c: Dict) -> str:
    profile = c.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
        profile.get("current_company", ""),
        c.get("_raw_text", ""),
        c.get("open_source", ""),
    ]
    for role in c.get("career_history", []):
        parts.extend([role.get("title", ""), role.get("description", ""), role.get("company", "")])
    for skill in c.get("skills", []):
        name = skill.get("name", "") if isinstance(skill, dict) else str(skill)
        parts.append(name)
    for edu in c.get("education", []):
        if isinstance(edu, dict):
            parts.extend([
                edu.get("degree", ""),
                edu.get("field", ""),
                edu.get("school", ""),
                edu.get("institution", ""),
                edu.get("major", ""),
                edu.get("specialization", "")
            ])
        else:
            parts.append(str(edu))
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", "") if isinstance(cert, dict) else str(cert))
    for proj in c.get("projects", []):
        if isinstance(proj, dict):
            parts.extend([proj.get("name", ""), proj.get("description", "")])
        else:
            parts.append(str(proj))
    for pub in c.get("publications", []):
        if isinstance(pub, dict):
            parts.append(pub.get("title", ""))
        else:
            parts.append(str(pub))
    return " ".join(filter(None, parts))

def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def normalize_bm25_scores(raw_scores: List[float]) -> List[float]:
    """Percentile-based BM25 normalization (95th percentile ceiling)."""
    if not raw_scores or all(s <= 0 for s in raw_scores):
        return [0.0] * len(raw_scores)
    positive = sorted(s for s in raw_scores if s > 0)
    if not positive:
        return [0.0] * len(raw_scores)
    p95_idx = max(0, int(len(positive) * 0.95) - 1)
    ceiling = positive[p95_idx] or positive[-1]
    return [min(1.0, s / ceiling) for s in raw_scores]

def days_since(date_str: str) -> int:
    try:
        return (TODAY - datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()).days
    except:
        return 9999

def exp_score(years: float, min_years: int, max_years: int) -> float:
    if min_years <= years <= max_years:
        return 1.0
    elif years < min_years:
        gap = min_years - years
        return max(0.15, 1.0 - gap * 0.12)
    else:
        gap = years - max_years
        return max(0.72, 1.0 - gap * 0.03)

def title_relevance(title: str, preferred_titles: List[str], red_flag_bgs: List[str]) -> float:
    t = title.lower()
    for rf in red_flag_bgs:
        if rf.lower() in t: return 0.05
    for pt in preferred_titles:
        if pt.lower() in t: return 1.0
    if any(w in t for w in ["engineer", "developer", "scientist", "architect", "lead", "head"]): return 0.85
    if any(w in t for w in ["analyst", "specialist", "consultant"]): return 0.55
    if any(w in t for w in ["manager", "coordinator"]): return 0.45
    return 0.35

SKILL_ALIASES: Dict[str, List[str]] = {
    "python": ["py", "python3"],
    "javascript": ["js", "es6", "es2015", "ecmascript"],
    "typescript": ["ts"],
    "kubernetes": ["k8s"],
    "tensorflow": ["tf"],
    "pytorch": ["torch"],
    "postgresql": ["postgres", "pg", "psql"],
    "elasticsearch": ["elastic", "es"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "large language model": ["llm", "llms"],
    "retrieval augmented generation": ["rag"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp"],
    "microsoft azure": ["azure"],
    "continuous integration": ["ci", "ci/cd"],
    "continuous deployment": ["cd", "ci/cd"],
    "react": ["reactjs", "react.js"],
    "node": ["nodejs", "node.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs"],
    "go": ["golang"],
    "c++": ["cpp", "c plus plus"],
    "c#": ["csharp", "dotnet", ".net"],
    "huggingface": ["hugging face", "hf"],
}

_ALIAS_MAP: Dict[str, str] = {}
for _canonical, _aliases in SKILL_ALIASES.items():
    for _a in _aliases:
        _ALIAS_MAP[_a] = _canonical

_WHOLE_WORD_ONLY = {"go", "r", "c", "js", "ts", "py", "ml", "ai", "dl", "nlp", "sql"}

def _normalize_skill(s: str) -> str:
    s = s.lower().strip()
    return _ALIAS_MAP.get(s, s)

def _skill_match(skill_l: str, skill_names: set, text_lower: str) -> float:
    canon = _normalize_skill(skill_l)
    normalized_names = {_normalize_skill(sn) for sn in skill_names}
    if canon in normalized_names:
        return 1.0
    if len(canon) > 3 and canon not in _WHOLE_WORD_ONLY:
        for sn in normalized_names:
            if (canon in sn or sn in canon) and len(min(canon, sn, key=len)) > 3:
                return 1.0
    if canon.endswith('s') or canon in ("go", "r", "c", "js", "ts", "py"):
        pattern = r'\b' + re.escape(canon) + r'\b'
    else:
        pattern = r'\b' + re.escape(canon) + r's?\b'
        
    if re.search(pattern, text_lower):
        return 0.6
        
    if canon != skill_l:
        if skill_l.endswith('s') or skill_l in ("go", "r", "c", "js", "ts", "py"):
            pattern_raw = r'\b' + re.escape(skill_l) + r'\b'
        else:
            pattern_raw = r'\b' + re.escape(skill_l) + r's?\b'
        if re.search(pattern_raw, text_lower):
            return 0.6
    return 0.0

def skill_alignment_score(candidate_skills: List, must_have: List[str], nice_to_have: List[str], full_text: str) -> Tuple[float, List[str]]:
    skill_names = set()
    for s in candidate_skills:
        raw = s.get("name", "") if isinstance(s, dict) else str(s)
        skill_names.add(raw.lower().strip())

    text_lower = full_text.lower()
    matched = []
    score = 0.0

    for skill in must_have:
        skill_l = skill.lower().strip()
        if not skill_l: continue
        m = _skill_match(skill_l, skill_names, text_lower)
        if m >= 1.0:
            score += 3.0
            matched.append(skill)
        elif m >= 0.6:
            score += 1.5
            matched.append(f"~{skill}")

    for skill in nice_to_have:
        skill_l = skill.lower().strip()
        if not skill_l: continue
        m = _skill_match(skill_l, skill_names, text_lower)
        if m >= 1.0:
            score += 1.0
            matched.append(skill)
        elif m >= 0.6:
            score += 0.5
            matched.append(f"~{skill}")

    jd_kw_bonus = 0.0
    for sn in skill_names:
        if len(sn) > 3 and re.search(r'\b' + re.escape(sn) + r'\b', text_lower):
            jd_kw_bonus += 0.15
    jd_kw_bonus = min(jd_kw_bonus, 1.5)

    max_possible = max(1.0, len(must_have) * 3.0 + len(nice_to_have) * 1.0 + 1.5)
    return min(1.0, (score + jd_kw_bonus) / max_possible), matched[:12]

def production_signal_score(text: str) -> float:
    text_l = text.lower()
    STRONG_POS = [
        r"shipped", r"production", r"deployed", r"deployment", r"scaling", r"scaled",
        r"a/b test", r"launched", r"released", r"migrated", r"real-time", r"realtime",
        r"\d+m\s+users", r"million users", r"billion", r"latency", r"monitoring",
        r"serving", r"inference at scale", r"reduced .{0,30}by \d+", r"improved .{0,30}by \d+",
        r"optimized", r"processing \d+",
    ]
    WEAK_POS = ["built", "implemented", "integrated", "automated", "pipeline",
                "feature", "model in production", "end-to-end", "end to end"]
    NEG = ["research only", "no production", "theoretical", "academic only", "arxiv", "published paper"]

    pos_strong = sum(1 for p in STRONG_POS if re.search(p, text_l))
    pos_weak   = sum(1 for p in WEAK_POS   if p in text_l)
    neg        = sum(1 for n in NEG         if n in text_l)

    net = pos_strong * 2.0 + pos_weak * 0.5 - neg * 2.0
    if net <= 0 and pos_strong == 0 and pos_weak == 0:
        return 0.45
    return max(0.0, min(1.0, net / 12.0))

def company_type_score(company: str, preferred_companies: List[str], jd_text: str) -> float:
    c = company.lower()
    jd_l = jd_text.lower()
    service_cos = {"tcs", "infosys", "wipro", "accenture", "cognizant", "hcl", "capgemini",
                   "tech mahindra", "mphasis", "ltimindtree", "hexaware"}
    known_product = {"google", "amazon", "microsoft", "meta", "apple", "netflix", "uber",
                     "airbnb", "stripe", "razorpay", "swiggy", "zomato", "meesho", "zepto",
                     "cred", "phonepe", "paytm", "freshworks", "zoho", "browserstack"}
    prefers_product = "product company" in jd_l or "not service" in jd_l or "not consulting" in jd_l
    prefers_startup = "startup" in jd_l or "fast-growing" in jd_l or "early stage" in jd_l
    if any(sc in c for sc in service_cos): return 0.3 if prefers_product else 0.6
    if any(pc in c for pc in known_product): return 1.0
    return 0.72

def engagement_score(signals: Dict) -> float:
    rrr = float(signals.get("recruiter_response_rate", 0.7))
    icr = float(signals.get("interview_completion_rate", 0.7))
    open_flag = 1.04 if signals.get("open_to_work_flag", True) else 1.0
    return min(1.15, (0.65 + 0.35 * rrr) * (0.95 + 0.05 * icr) * open_flag)

def notice_period_score(signals: Dict, jd_analysis: Dict = None) -> float:
    np_days = int(signals.get("notice_period_days", 60))
    pref = (jd_analysis or {}).get("notice_period_preference_days", 30)
    # Score relative to what recruiter wants
    if np_days <= 0:    return 1.0
    elif np_days <= pref * 0.5:  return 0.98
    elif np_days <= pref:        return 0.88
    elif np_days <= pref * 1.5:  return 0.68
    elif np_days <= pref * 2.0:  return 0.48
    elif np_days <= 90:          return 0.32
    else:                        return 0.18

def github_oss_score(candidate: Dict, signals: Dict) -> Tuple[float, bool]:
    """Returns (score, has_data)."""
    gh = signals.get("github_activity_score", -1)
    oss = candidate.get("open_source", "")
    pubs = candidate.get("publications", [])

    has_data = (gh is not None and gh > 0) or bool(oss) or bool(pubs)
    if not has_data:
        return 0.0, False

    score = 0.0
    if gh is not None and gh > 0:
        score += min(0.6, gh / 100.0 * 0.6)
    if oss:
        score += 0.25
    if pubs:
        score += min(0.15, len(pubs) * 0.05)
    return min(1.0, score), True

def notice_period_bonus(signals: Dict) -> float:
    """Legacy micro-bonus (open-to-work uplift). Kept for backward compat."""
    return 0.01 if signals.get("open_to_work_flag", True) else 0.0

def github_bonus(signals: Dict) -> float:
    gh = signals.get("github_activity_score", -1)
    if gh is not None and gh > 0:
        return min(0.06, gh / 100.0 * 0.06)
    return 0.0

# ──────────────────────────────────────────────────────────────
# Phase 8 — Confidence Engine
# ──────────────────────────────────────────────────────────────
def compute_confidence(
    skill_s: float, exp_s: float, available_signals: Dict[str, bool],
    jd_analysis: Dict, matched_skills: List[str]
) -> float:
    """
    Confidence = how much data we had to work with, how well it matched.
    Falls when data is missing, but does not affect the fair score.
    """
    jd_clarity = 0.0 if jd_analysis.get("_using_fallback") else 1.0
    data_completeness = sum(1 for v in available_signals.values() if v) / max(1, len(available_signals))
    skill_coverage = min(1.0, len([s for s in matched_skills if not s.startswith("~")]) /
                         max(1, len(jd_analysis.get("must_have_skills", ["x"]))))

    raw = (
        0.30 * skill_s
        + 0.20 * exp_s
        + 0.25 * jd_clarity
        + 0.15 * data_completeness
        + 0.10 * skill_coverage
    )
    return round(min(0.99, max(0.30, raw)), 4)

# ──────────────────────────────────────────────────────────────
# Phase 7 — Recruiter Explanation Engine
# ──────────────────────────────────────────────────────────────
def build_reasoning(info: Dict, jd_analysis: Dict, next_candidate: Dict = None) -> str:
    """Generate recruiter-grade reasoning with strengths, concerns, and comparisons."""
    parts = []
    bd = info["breakdown"]

    # ── Strengths ─────────────────────────────────────────────
    strengths = []
    if bd["skill_alignment"]    >= 0.75: strengths.append(f"Strong skill match ({bd['skill_alignment']*100:.0f}%)")
    if bd["experience_fit"]     >= 0.80: strengths.append(f"Experience in range ({bd['experience_fit']*100:.0f}%)")
    if bd["career_progression"] >= 0.70: strengths.append("Strong career trajectory")
    if bd["notice_period"]      >= 0.85: strengths.append("Quick joiner")
    if bd["production_signals"] >= 0.70: strengths.append("Proven production experience")
    if bd.get("github_oss", 0)  >= 0.50: strengths.append("Active open-source presence")
    if info["signals"].get("github_score", -1) > 60: strengths.append("Strong GitHub activity")
    # Layer-level strengths
    if bd.get("technical_fit", 0) >= 0.78: strengths.append(f"High technical fit ({bd['technical_fit']*100:.0f}%)")
    if bd.get("career_fit", 0)    >= 0.78: strengths.append(f"Strong career fit ({bd['career_fit']*100:.0f}%)")

    # ── Concerns ──────────────────────────────────────────────
    concerns = []
    if bd["skill_alignment"]    <  0.40: concerns.append("Low skill alignment")
    if bd["experience_fit"]     <  0.60: concerns.append("Experience mismatch")
    if bd["career_progression"] <  0.40: concerns.append("Unstable career path")
    if bd["notice_period"]      <  0.40: concerns.append(f"Long notice period ({info['signals'].get('notice_days', 60)}d)")
    if bd["company_fit"]        <  0.40: concerns.append("Service company background")
    if bd.get("recruiter_fit", 0) < 0.45: concerns.append("Low recruiter signal quality")

    if strengths:
        parts.append("✓ " + "; ".join(strengths[:3]))
    if concerns:
        parts.append("• " + "; ".join(concerns[:2]))

    # ── Comparative justification ──────────────────────────────
    if next_candidate:
        nbd = next_candidate["breakdown"]
        comp_parts = []
        if bd["experience_fit"] > nbd["experience_fit"] + 0.03 and bd["skill_alignment"] < nbd["skill_alignment"]:
            diff = int((bd["experience_fit"] - nbd["experience_fit"]) * 100)
            comp_parts.append(f"Ranked higher: superior experience fit (+{diff}%) outweighs minor skill gap.")
        elif bd["career_progression"] > nbd["career_progression"] + 0.10:
            comp_parts.append("Stronger career stability and growth trajectory.")
        elif bd["notice_period"] > nbd["notice_period"] + 0.15:
            comp_parts.append("Faster availability is a meaningful differentiator here.")
        elif bd["skill_alignment"] > nbd["skill_alignment"] + 0.05:
            diff = int((bd["skill_alignment"] - nbd["skill_alignment"]) * 100)
            comp_parts.append(f"Better skill alignment (+{diff}%) is the deciding factor.")
        if comp_parts:
            parts.append(" | ".join(comp_parts))

    if len(parts) < 2:
        if bd["skill_alignment"] > 0.85:
            parts.append("Exceptional skill coverage for this role.")
        elif bd["career_progression"] > 0.80:
            parts.append("Outstanding career trajectory with clear progression.")

    if not parts:
        parts.append("Profile reviewed for technical alignment.")

    return " ".join(parts)

# ──────────────────────────────────────────────────────────────
# Phase 5 — Three-Layer Scoring + Phase 4 Adaptive Redistribution
# ──────────────────────────────────────────────────────────────
def score_candidate(
    c: Dict,
    bm25_norm: float,
    bm25_max: float,   # kept for API compat, unused
    jd_analysis: Dict,
    jd_text: str,
    jd_weights: Dict[str, float] = None,
) -> Tuple[float, Dict]:
    """
    Core scoring function — adaptive, three-layer, explainable.
    Three layers:
      - Technical Fit  (skills, semantic, production, BM25, certs)
      - Career Fit     (experience, career progression, company, title)
      - Recruiter Fit  (notice period, engagement, open-to-work, github)
    Final = weighted sum of all signals using dynamically redistributed weights.
    """
    c_filtered = apply_fairness_filter(c)
    profile   = c_filtered.get("profile", {})
    signals   = c_filtered.get("redrob_signals", {})
    skills    = c_filtered.get("skills", [])
    career    = c_filtered.get("career_history", [])
    full_text = build_candidate_text(c_filtered)

    # ── Generate per-JD weights (once, cached by caller ideally) ──
    if jd_weights is None:
        jd_weights = generate_jd_weights(jd_analysis, jd_text)

    # ═══════════════════════════════════════════════════════════
    # COMPUTE ALL RAW SIGNALS
    # ═══════════════════════════════════════════════════════════

    # 1. Skill alignment
    skill_s, matched_skills = skill_alignment_score(
        skills, jd_analysis.get("must_have_skills", []),
        jd_analysis.get("nice_to_have_skills", []), full_text,
    )

    # 2. Experience fit
    years   = float(profile.get("years_of_experience", 0))
    exp_s   = exp_score(years, jd_analysis.get("years_min", 2), jd_analysis.get("years_max", 10))

    # 3. Career progression
    career_s, career_detail = analyze_career_progression(career)
    has_career = bool(career)

    # 4. Semantic similarity
    if HAS_TRANSFORMERS:
        jd_emb   = bi_encoder.encode(jd_text[:2000], convert_to_tensor=True)
        cand_emb = bi_encoder.encode(full_text[:2000], convert_to_tensor=True)
        cos_sim  = F.cosine_similarity(jd_emb, cand_emb, dim=0).item()
        cross_raw  = cross_encoder.predict([jd_text[:1000], full_text[:1000]])
        cross_norm = float(1 / (1 + math.exp(-cross_raw)))
        semantic_s = 0.40 * cos_sim + 0.60 * cross_norm
    else:
        jd_tokens_set   = set(re.findall(r"[a-z][a-z0-9]{2,}", jd_text.lower()))
        cand_tokens_set = set(re.findall(r"[a-z][a-z0-9]{2,}", full_text.lower()))
        if jd_tokens_set or cand_tokens_set:
            intersection = len(jd_tokens_set & cand_tokens_set)
            union        = len(jd_tokens_set | cand_tokens_set)
            semantic_s   = min(1.0, (intersection / union if union else 0.0) * 4.0)
        else:
            semantic_s = 0.0

    # 5. Production signals
    ship_s = production_signal_score(full_text)

    # 6. Company fit
    company_s = company_type_score(
        profile.get("current_company", ""),
        jd_analysis.get("preferred_companies", []), jd_text,
    )

    # 7. BM25
    bm25_s = float(bm25_norm)

    # 8. Notice period
    notice_s = notice_period_score(signals, jd_analysis)

    # 9. GitHub / OSS
    github_s, has_github = github_oss_score(c_filtered, signals)

    # 10. Education modifier
    edu_mod, has_edu = parse_and_score_education(full_text, jd_analysis)

    # 11. Engagement (open-to-work, response rate)
    eng = engagement_score(signals)

    # 12. Title relevance (used inside career layer)
    title_s = title_relevance(
        profile.get("current_title", ""),
        jd_analysis.get("preferred_titles", []),
        jd_analysis.get("red_flag_backgrounds", []),
    )

    # ═══════════════════════════════════════════════════════════
    # PHASE 4 — Detect available signals per candidate
    # ═══════════════════════════════════════════════════════════
    available_signals = {
        "skill_alignment":    skill_s > 0 or bool(skills),
        "experience_fit":     years > 0,
        "career_progression": has_career,
        "semantic_match":     True,   # always computable
        "production_signals": True,   # text-based, always computable
        "notice_period":      "notice_period_days" in signals,
        "company_fit":        bool(profile.get("current_company")),
        "bm25_match":         True,
        "education":          has_edu,
        "github_oss":         has_github,
        "title_relevance":    bool(profile.get("current_title")),
        "certifications":     bool(c_filtered.get("certifications")),
    }

    # Redistribute weights for this candidate
    effective_weights = redistribute_weights(jd_weights, available_signals)

    # ═══════════════════════════════════════════════════════════
    # PHASE 5 — Three-Layer Scoring
    # ═══════════════════════════════════════════════════════════
    # Gather each signal's effective weight
    w = effective_weights

    # ── Technical Fit ─────────────────────────────────────────
    # Signals: skills, semantic, production, bm25, github/oss
    tech_total_w = w["skill_alignment"] + w["semantic_match"] + w["production_signals"] + w["bm25_match"] + w["github_oss"]
    if tech_total_w > 0:
        technical_fit = (
            w["skill_alignment"]    * skill_s
            + w["semantic_match"]   * semantic_s
            + w["production_signals"] * ship_s
            + w["bm25_match"]       * bm25_s
            + w["github_oss"]       * github_s
        ) / tech_total_w
    else:
        technical_fit = 0.5

    # ── Career Fit ────────────────────────────────────────────
    # Signals: experience, career_progression, company, education (as direct score here)
    edu_score = min(1.0, max(0.0, edu_mod - 0.5))   # convert modifier to [0,1] score
    career_total_w = w["experience_fit"] + w["career_progression"] + w["company_fit"] + w["education"] + w["title_relevance"]
    if career_total_w > 0:
        career_fit = (
            w["experience_fit"]     * exp_s
            + w["career_progression"] * (career_s if has_career else exp_s * 0.5)
            + w["company_fit"]      * company_s
            + w["education"]        * edu_score
            + w["title_relevance"] * title_s
        ) / career_total_w
    else:
        career_fit = 0.5

    # ── Recruiter Fit ─────────────────────────────────────────
    # Signals: notice_period + engagement (open-to-work, response rate)
    recruiter_fit_raw = notice_s * 0.7 + (eng - 1.0 + 0.5) * 0.3   # eng is ~[0.65, 1.15]
    recruiter_fit = min(1.0, max(0.0, recruiter_fit_raw))

    # ── Combine layers with dynamic layer weights ─────────────
    # Layer weights derived from signal weights: tech signals / total, etc.
    layer_tech_w   = tech_total_w
    layer_career_w = career_total_w
    layer_rec_w    = w["notice_period"]   # recruiter layer anchored to notice weight

    layer_total = layer_tech_w + layer_career_w + layer_rec_w
    if layer_total > 0:
        base = (
            layer_tech_w   / layer_total * technical_fit
            + layer_career_w / layer_total * career_fit
            + layer_rec_w    / layer_total * recruiter_fit
        )
    else:
        base = (technical_fit + career_fit + recruiter_fit) / 3.0

    # Education as multiplicative modifier (PhD for PhD role → small uplift)
    base = max(0.0, min(1.0, base * edu_mod))

    # Micro-bonuses: open-to-work, github
    gh_b     = github_bonus(signals)
    notice_b = notice_period_bonus(signals)
    final_0to1 = min(1.0, base * eng + gh_b + notice_b)

    # Hard floor: matched nothing meaningful → cap at 18%
    if skill_s < 0.05 and bm25_s < 0.05 and semantic_s < 0.25:
        final_0to1 = min(final_0to1, 0.18)

    final = round(final_0to1, 4)

    # ═══════════════════════════════════════════════════════════
    # PHASE 8 — Confidence Score
    # ═══════════════════════════════════════════════════════════
    confidence = compute_confidence(skill_s, exp_s, available_signals, jd_analysis, matched_skills)

    # ── Breakdown (all 0–1, frontend ×100 for display) ────────
    breakdown = {
        # Core signals
        "skill_alignment":    round(skill_s,       4),
        "experience_fit":     round(exp_s,          4),
        "career_progression": round(career_s,       4),
        "notice_period":      round(notice_s,       4),
        "semantic_match":     round(semantic_s,     4),
        "bm25_match":         round(bm25_s,         4),
        "production_signals": round(ship_s,         4),
        "company_fit":        round(company_s,      4),
        "title_relevance":    round(title_s,        4),
        "engagement":         round(eng,            4),
        "github_oss":         round(github_s,       4),
        # Three-layer scores
        "technical_fit":      round(technical_fit,  4),
        "career_fit":         round(career_fit,     4),
        "recruiter_fit":      round(recruiter_fit,  4),
        # Dynamic weights used (for explainability dashboard)
        "_weights":           {k: round(v, 4) for k, v in effective_weights.items()},
        "_role_type":         jd_weights.get("_role_type", "generic"),
        "_available":         available_signals,
    }

    raw_name = c.get("profile", {}).get("anonymized_name") or c.get("name", "Unknown")

    info = {
        "candidate_id":   c["candidate_id"],
        "final_score":    final,
        "confidence_score": confidence,
        "name":           raw_name,
        "current_title":  profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "years_exp":      years,
        "location":       profile.get("location", ""),
        "matched_skills": matched_skills,
        "breakdown":      breakdown,
        "career_detail":  career_detail,
        "signals": {
            "open_to_work":   signals.get("open_to_work_flag", True),
            "github_score":   signals.get("github_activity_score", -1),
            "response_rate":  signals.get("recruiter_response_rate", 0),
            "notice_days":    signals.get("notice_period_days", 60),
            "completeness":   signals.get("profile_completeness_score", 0),
            "last_active_days": days_since(signals.get("last_active_date", "2024-06-01")),
        },
    }
    return final_0to1, info

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
        jd_analysis = await analyze_jd_with_ai(jd_text)
    else:
        jd_analysis = extract_jd_heuristic(jd_text)
        jd_analysis["_using_fallback"] = True

    # Step 2: Generate dynamic weights ONCE per JD
    jd_weights = generate_jd_weights(jd_analysis, jd_text)
    jd_analysis["_role_type"] = jd_weights.get("_role_type", "generic")
    jd_analysis["_dynamic_weights"] = {k: v for k, v in jd_weights.items() if not k.startswith("_")}

    # Step 3: BM25 tokens
    jd_tokens = tokenize_for_bm25(jd_text)
    for kw in jd_analysis.get("domain_keywords", []):
        jd_tokens.extend(tokenize_for_bm25(kw))
    for skill in jd_analysis.get("must_have_skills", []):
        jd_tokens.extend(tokenize_for_bm25(skill))

    # Step 4: Normalize candidates
    candidates = [normalize_candidate(c) for c in raw_candidates]

    # Step 5: BM25 index
    if HAS_BM25:
        corpus = [tokenize_for_bm25(build_candidate_text(c)) for c in candidates]
        valid_pairs = [(i, doc) for i, doc in enumerate(corpus) if doc]
        if valid_pairs:
            valid_indices, valid_docs = zip(*valid_pairs)
            bm25_model = BM25Okapi(valid_docs)
            raw_scores_valid = list(bm25_model.get_scores(jd_tokens))
            bm25_raw_all = [0.0] * len(candidates)
            for idx, score in zip(valid_indices, raw_scores_valid):
                bm25_raw_all[idx] = float(score)
            bm25_scores = normalize_bm25_scores(bm25_raw_all)
        else:
            bm25_scores = [0.0] * len(candidates)
    else:
        bm25_scores = [0.0] * len(candidates)

    # Step 6: Score all candidates (pass pre-computed jd_weights)
    scored = []
    for i, c in enumerate(candidates):
        final, info = score_candidate(c, bm25_scores[i], 1.0, jd_analysis, jd_text, jd_weights)
        scored.append((final, info))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    # Step 7: Build output
    results = []
    for i, (score, info) in enumerate(top):
        rank = i + 1
        next_cand_info = top[i + 1][1] if i + 1 < len(top) else None
        reasoning = build_reasoning(info, jd_analysis, next_candidate=next_cand_info)

        bd = info["breakdown"]
        results.append({
            "rank":             rank,
            "candidate_id":     info["candidate_id"],
            "score":            info["final_score"],
            "confidence":       info["confidence_score"],
            "name":             info["name"],
            "current_title":    info["current_title"],
            "current_company":  info["current_company"],
            "years_experience": info["years_exp"],
            "location":         info["location"],
            "skills_matched":   info["matched_skills"],
            "reasoning":        reasoning,
            "score_breakdown":  bd,
            "signals":          info["signals"],
        })

    return results, jd_analysis

# ──────────────────────────────────────────────────────────────
# API Endpoints (unchanged shape — frontend compatibility)
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
    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        jd_content = parse_jd_file(jd_bytes, jd_file.filename)
    elif jd_text:
        jd_content = jd_text
    else:
        raise HTTPException(400, "Provide job description as file or text")

    if not jd_content.strip():
        raise HTTPException(400, "Job description is empty")

    if candidates_file and candidates_file.filename:
        cand_bytes = await candidates_file.read()
        raw_candidates = parse_candidates_file(cand_bytes, candidates_file.filename)
    elif candidates_text:
        raw_candidates = json.loads(candidates_text)
    else:
        raise HTTPException(400, "Provide candidates as file or text")

    if not raw_candidates:
        raise HTTPException(400, "Candidate pool is empty")

    results, jd_analysis = await run_ranking_pipeline(jd_content, raw_candidates, top_n, use_ai)

    return {
        "success": True,
        "total_candidates": len(raw_candidates),
        "shortlisted": len(results),
        "jd_analysis": jd_analysis,
        "results": results,
        "fair_ranking_enabled": True,
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
        "rank", "candidate_id", "name", "score_%", "confidence_%", "current_title",
        "current_company", "years_experience", "location",
        "skills_matched", "reasoning",
        # Core signal breakdown
        "skill_alignment_%", "experience_fit_%", "career_progression_%", "notice_period_%",
        "semantic_match_%", "bm25_match_%", "production_signals_%", "company_fit_%",
        "title_relevance_%", "engagement_%", "github_oss_%",
        # Three-layer scores
        "technical_fit_%", "career_fit_%", "recruiter_fit_%",
    ])
    for r in results:
        bd = r.get("score_breakdown", {})
        def pct(v): return f"{round(v * 100, 1)}" if v is not None else ""
        writer.writerow([
            r.get("rank"), r.get("candidate_id"), r.get("name"),
            pct(r.get("score")), pct(r.get("confidence")),
            r.get("current_title"), r.get("current_company"),
            r.get("years_experience"), r.get("location"),
            "; ".join(r.get("skills_matched", [])),
            r.get("reasoning"),
            pct(bd.get("skill_alignment")), pct(bd.get("experience_fit")),
            pct(bd.get("career_progression")), pct(bd.get("notice_period")),
            pct(bd.get("semantic_match")), pct(bd.get("bm25_match")),
            pct(bd.get("production_signals")), pct(bd.get("company_fit")),
            pct(bd.get("title_relevance")), pct(bd.get("engagement")),
            pct(bd.get("github_oss")),
            pct(bd.get("technical_fit")), pct(bd.get("career_fit")), pct(bd.get("recruiter_fit")),
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ranked_candidates.csv"},
    )

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "bm25_available": HAS_BM25,
        "transformers_available": HAS_TRANSFORMERS,
        "version": "3.0.0",
    }

from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

static_dir = Path(__file__).parent.parent / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")