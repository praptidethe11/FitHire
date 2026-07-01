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
import torch
import math
from datetime import date, datetime
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    HAS_TRANSFORMERS = True
    device = "cpu"
    # Force torch to use all available CPU cores for inference. Left
    # unset, torch sometimes defaults to a small thread count (or 1) on
    # Windows depending on how MKL/OMP env vars were inherited from the
    # shell, which silently makes CPU inference several times slower
    # with no visible error -- just a quiet, massive slowdown.
    import multiprocessing as _mp
    torch.set_num_threads(_mp.cpu_count())
    print(f"[INFO] torch using {torch.get_num_threads()} CPU threads "
          f"(of {_mp.cpu_count()} available cores)")
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
except ImportError:
    # sentence-transformers / torch simply aren't installed.
    HAS_TRANSFORMERS = False
    print(
        "[WARN] Semantic models unavailable — sentence-transformers/torch are "
        "not installed. Run `pip install -r requirements.txt`, then "
        "`python download_models.py` once, then re-run. Falling back to the "
        "non-semantic heuristic scorer for now."
    )
except OSError as e:
    # Model isn't in the local HuggingFace cache. This is the expected
    # failure mode when HF_HUB_OFFLINE=1 is set (see rank_cli.py) and
    # download_models.py hasn't been run yet — huggingface_hub raises an
    # OSError/LocalEntryNotFoundError instead of trying the network.
    # Fail LOUD here (not silently) so it's obvious in logs why semantic
    # scoring is unavailable, even though the pipeline still degrades
    # gracefully to the heuristic fallback rather than crashing outright.
    HAS_TRANSFORMERS = False
    print(
        "[WARN] Semantic models unavailable — could not load from the local "
        "HuggingFace cache (offline mode is likely on and the models were "
        "never downloaded). Run `python download_models.py` once, then "
        "re-run. Falling back to the non-semantic heuristic scorer for now.\n"
        f"        Underlying error: {e}",
    )

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

HAS_OCR = True

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
    # Use the full JD text, not just the first 1500 chars. This runs once per
    # JD (not per-candidate), so the extra regex/`in` scans over the full text
    # cost nothing meaningful — but truncating was silently dropping keyword
    # hits that only appear later in the JD (e.g. a "skills inventory" or
    # "what we're looking for" section past the intro), which caused clearly
    # domain-specific JDs to fall back to "generic" and lose their role-type
    # amplifiers entirely.
    combined = f"{title} {must} {text}"

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
_DEGREE_TIERS = {
    "phd": 4, "doctorate": 4, "master": 3, "ms": 3, "msc": 3,
    "mba": 3, "mtech": 3, "bachelor": 2, "bs": 2, "bsc": 2,
    "btech": 2, "ba": 2, "diploma": 1, "associate": 1
}
# Precompiled once at import time since the degree vocabulary is fixed, not derived
# per-candidate or per-JD — avoids recompiling the same 14 patterns for every candidate.
_DEGREE_TIER_PATTERNS = [(re.compile(r'\b' + re.escape(d) + r'\b'), t) for d, t in _DEGREE_TIERS.items()]

def parse_and_score_education(candidate_text: str, jd_analysis: Dict) -> Tuple[float, bool]:
    """Returns (modifier_float, has_education_data)."""
    text = candidate_text.lower()
    cand_tier = 0
    for pattern, tier in _DEGREE_TIER_PATTERNS:
        if pattern.search(text):
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

# ==============================================================================
# OPTIMIZED SEMANTIC SCORING PIPELINE FOR HACKATHON COMPLIANCE
# ==============================================================================

def precompute_semantic_scores(
    candidates: List[Dict[Any, Any]], 
    jd_text: str, 
    top_indices: List[int], 
    deadline: float,
    bi_encoder_name: str = "all-MiniLM-L6-v2",
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    target_threads: int = 4
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Computes semantic scores using pre-downloaded bi-encoders and cross-encoders.
    Uses micro-batching and rigorous torch inference isolation to keep execution 
    well under the 300s wall-clock hackathon budget.
    """
    bi_scores = {}
    cross_scores = {}
    
    if not HAS_TRANSFORMERS:
        print("[WARN] Transformers not available. Falling back to heuristic scoring.")
        return bi_scores, cross_scores

    # Configure and verify CPU parallelism parameters
    torch.set_num_threads(target_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
        
    print(f"[DEBUG] Active PyTorch Threads immediately before loading: {torch.get_num_threads()}")

    try:
        # Load models explicitly to CPU with offline weights
        print("[AI Setup] Loading Bi-Encoder model to CPU...")
        bi_model = SentenceTransformer(bi_encoder_name, device="cpu")
        bi_model.eval()
        
        print("[AI Setup] Loading Cross-Encoder model to CPU...")
        cross_model = CrossEncoder(cross_encoder_name, device="cpu")
        cross_model.model.eval()  # Force underlying PyTorch NN to evaluation mode
    except Exception as e:
        print(f"[ERROR] Failed to load local weights: {e}. Falling back to heuristics.")
        return bi_scores, cross_scores

    # Isolate all calculations inside inference mode to eliminate gradient tracking overhead
    with torch.inference_mode(), torch.no_grad():
        print(f"[DEBUG] Executing inference with threads: {torch.get_num_threads()}")
        
        # --- PHASE 1: Bi-Encoder Embeddings ---
        if time.time() >= deadline:
            print("[DEADLINE] No budget left for Bi-Encoder phase.")
            return bi_scores, cross_scores
            
        print("[AI Inference] Computing Bi-Encoder match scores...")
        jd_embedding = bi_model.encode(jd_text, convert_to_tensor=True, show_progress_bar=False)
        
        # Use a localized micro-batch size of 64 to optimize L3 cache reuse
        bi_batch_size = 64
        for i in range(0, len(top_indices), bi_batch_size):
            if time.time() >= deadline:
                print(f"[DEADLINE] Bi-Encoder computation truncated at index {i}/{len(top_indices)}")
                break
                
            batch_idxs = top_indices[i : i + bi_batch_size]
            batch_texts = [candidates[idx].get('cached_text', '') for idx in batch_idxs]
            
            cand_embeddings = bi_model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
            # Compute cosine similarities via dot product on normalized tensors
            similarities = torch.nn.functional.cosine_similarity(jd_embedding.unsqueeze(0), cand_embeddings)
            
            for idx, sim in zip(batch_idxs, similarities.tolist()):
                bi_scores[idx] = float(sim)

        # --- PHASE 2: Cross-Encoder High-Fidelity Scoring ---
        print("[AI Inference] Computing Cross-Encoder re-ranking scores...")
        # Reduce batch size to 32 for highly granular deadline checks and lower memory footprint
        cross_batch_size = 32
        
        for i in range(0, len(top_indices), cross_batch_size):
            # Check deadline before processing each micro-batch
            if time.time() >= deadline:
                print(f"[DEADLINE] Cross-Encoder loop broke at index {i}/{len(top_indices)} to guarantee runtime bounds.")
                break
                
            batch_idxs = top_indices[i : i + cross_batch_size]
            
            # Construct sequence pairs: (Job Description, Candidate Experience Summary)
            pairs = [(jd_text, candidates[idx].get('cached_text', '')) for idx in batch_idxs]
            
            # Execute predictions directly through the evaluated PyTorch backend
            predictions = cross_model.predict(pairs, batch_size=cross_batch_size, show_progress_bar=False)
            
            for idx, score in zip(batch_idxs, predictions):
                cross_scores[idx] = float(score)

    return bi_scores, cross_scores

# Existing line in your main.py:
app = FastAPI(
    title="FitHire — Adaptive Recruiter Intelligence Engine",
    version="3.0.0"
)

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
def _ocr_image_bytes(file_bytes: bytes, filename: str = "") -> str:
    """
    Pure-Python text extractor. Strictly accepts text layers from PDF and TXT.
    Rejects images and unsupported binaries immediately.
    """
    filename_lower = filename.lower() if filename else ""

    # 1. Handle plain text files
    if filename_lower.endswith('.txt'):
        try:
            return file_bytes.decode('utf-8', errors='ignore').strip()
        except Exception as e:
            print(f"Error decoding text file: {e}")
            return ""

    # 2. Handle PDFs using pypdf text-layer extraction
    if filename_lower.endswith('.pdf'):
        if not HAS_PYPDF:
            print("Warning: pypdf is not installed — cannot extract PDF text. Install it with: pip install pypdf")
            return ""
        try:
            text_content = []
            with io.BytesIO(file_bytes) as open_pdf_file:
                reader = pypdf.PdfReader(open_pdf_file)
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text_content.append(extracted_text)

            final_text = "\n".join(text_content).strip()
            if not final_text:
                print("Warning: PDF contains no readable text layer (scanned/empty).")
            return final_text
        except Exception as e:
            print(f"Error parsing PDF file layers: {e}")
            return ""

    # 3. Explicitly reject images and all other extensions
    print(f"Unsupported file format: {filename}. Images (.png, .jpg) are not accepted.")
    return ""


def _parse_resume_text_to_candidate(text: str, ocr_confidence: float = 1.0, source_id: str = "") -> Dict:
    """
    Feature 4 & 5: Parse freeform resume text (from OCR or plain text) into
    a raw candidate dict, then normalize. Used for image/text resumes.
    The raw dict enters the same normalize_candidate pipeline as structured data.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Basic heuristic extraction from resume text
    raw: Dict[str, Any] = {
        "_raw_text": text,
        "_ocr_confidence": ocr_confidence,
        "candidate_id": source_id or f"RESUME_{abs(hash(text[:200]))}",
    }

    # Detect name: usually the first non-empty line before any section header
    if lines:
        first_line = lines[0]
        # Name heuristic: short, no numbers, title-case-ish
        if len(first_line) < 60 and not re.search(r"\d", first_line) and len(first_line.split()) <= 5:
            raw["name"] = first_line

    # Email extraction
    email_m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    if email_m:
        raw["email"] = email_m.group(0)

    # Phone extraction
    phone_m = re.search(r"[\+]?[\d][\d\s\-\(\)]{8,15}", text)
    if phone_m:
        raw["phone"] = phone_m.group(0).strip()

    # LinkedIn extraction
    linkedin_m = re.search(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
    if linkedin_m:
        raw["linkedin"] = "https://" + linkedin_m.group(0)

    # GitHub extraction
    github_m = re.search(r"github\.com/[\w\-]+", text, re.IGNORECASE)
    if github_m:
        raw["github"] = "https://" + github_m.group(0)

    # Use section detection to populate fields
    detected = _detect_sections_from_text(text)

    # Skills: collect skill-bearing lines and extract known skills
    skill_text = " ".join(detected.get("skills", []) + [text])
    raw["skills"] = _extract_skills_from_text(skill_text)

    # Summary: first paragraph before any career signal
    for i, line in enumerate(lines[:15]):
        if len(line) > 40 and not re.search(r"\b(20\d{2}|19\d{2})\b", line):
            raw["summary"] = line
            break

    # Experience / title inference: look for most senior-sounding title
    for line in lines:
        if re.search(r"\b(engineer|developer|scientist|architect|manager|lead|analyst|designer)\b", line.lower()):
            raw["current_title"] = line[:80]
            break

    # Career history from detected lines
    if detected.get("career"):
        raw["career_history"] = [{
            "title": raw.get("current_title", ""),
            "company": "",
            "description": "\n".join(detected["career"]),
            "duration_months": 0,
        }]

    # Education from detected lines
    if detected.get("education"):
        raw["education"] = [{"text": l} for l in detected["education"]]

    # Certifications
    if detected.get("certifications"):
        raw["certifications"] = [{"name": l} for l in detected["certifications"]]

    # Confidence penalty: if OCR confidence is below threshold, flag it
    if ocr_confidence < 0.5:
        raw["_low_ocr_confidence"] = True

    return raw


def parse_jd_file(file_bytes: bytes, filename: str) -> str:
    """
    Parse a JD file to plain text.
    - PDF/TXT: uses pure-Python pypdf via _ocr_image_bytes
    - DOCX: uses python-docx
    - Images: returns empty string gracefully (no Tesseract needed)
    """
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        # Route through _ocr_image_bytes which uses pypdf (pure-Python, no binaries)
        text = _ocr_image_bytes(file_bytes, filename=filename)
        if not text.strip():
            print(f"[JD] PDF text layer empty for '{filename}' (possibly a scanned image PDF)")
        return text

    elif ext == "docx":
        if not HAS_DOCX:
            raise HTTPException(400, "python-docx not installed — cannot parse DOCX job description")
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"[JD] DOCX parse error: {e}")
            return ""

    elif ext in ("jpg", "jpeg", "png"):
        # Images return empty — caller falls back to text-box input
        print(f"[JD] Image file '{filename}' received — no OCR binary available; use text box input")
        return ""

    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="replace")

    else:
        return file_bytes.decode("utf-8", errors="replace")

def parse_candidates_file(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Parse candidate data from any supported format.
    Feature 4: Image formats (JPG/PNG/JPEG) are OCR'd and parsed as resumes.
    Feature 5: Handles JSON, JSONL, CSV, Excel, and freeform image resumes.
    """
    ext = filename.lower().rsplit(".", 1)[-1]

    # ── Structured data formats ───────────────────────────────
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
            for key in ("candidates", "data", "results", "applicants", "profiles", "records"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]

    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_dict(orient="records")

    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(file_bytes))
        return df.to_dict(orient="records")

    # ── Feature 4: Image resume formats ──────────────────────
    elif ext in ("jpg", "jpeg", "png"):
        text = _ocr_image_bytes(file_bytes, filename=filename)
        if not text.strip():
            print(f"[WARN] Image extraction returned empty text for: {filename}")
            return []
        raw_candidate = _parse_resume_text_to_candidate(text, ocr_confidence=1.0, source_id=filename)
        return [raw_candidate]

    # ── Fallback: try JSON/JSONL text ─────────────────────────
    else:
        text = file_bytes.decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
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

# Heuristic JD parsing is the explicit primary design. AI analysis is removed.

def _extract_role_title(jd_text: str) -> str:
    """
    Pull the role title from the JD's opening lines instead of hard-coding
    "Software Engineer" for every JD. Looks for an explicit "Title:"/"Role:"/
    "Position:"/"Job Description:" label first, then falls back to the first
    non-empty line if it looks like a short title. This title feeds into
    _detect_role_type()'s keyword scan, so a wrong/generic title here was
    silently diluting role-type detection.
    """
    for line in jd_text.strip().split("\n")[:6]:
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"(?:job\s*)?(?:title|position|role|job\s*description)\s*[:\-]\s*(.+)",
            line, re.I,
        )
        candidate = m.group(1).strip() if m else line
        # Cut trailing " — Founding Team" / " | Company" / " - Company" suffixes
        candidate = re.split(r"\s+[–—-]\s+|\s*\|\s*", candidate)[0].strip()
        if 2 <= len(candidate) <= 80:
            return candidate
    return "Software Engineer"

def extract_jd_heuristic(jd_text: str) -> Dict:
    """Heuristic extraction from JD text — Phase 1 fallback."""
    text_lower = jd_text.lower()
    tokens = re.findall(r"[a-z0-9\+\#\.]+", text_lower)

    # ── Experience range ──────────────────────────────────────
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

    # ── Seniority ─────────────────────────────────────────────
    seniority = "mid"
    if any(w in text_lower for w in ["senior", "sr.", "lead", "principal", "staff"]):
        seniority = "senior"
    elif any(w in text_lower for w in ["junior", "jr.", "entry", "entry-level", "fresher", "graduate", "intern"]):
        seniority = "junior"
    elif any(w in text_lower for w in ["manager", "head of", "director", "vp"]):
        seniority = "manager"

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
        "role_title": _extract_role_title(jd_text),
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
# Feature 1: Universal Field Name Normalization
# Feature 3: Experience Inference from Role Labels
# Feature 6: Synonym Recognition
# Feature 7: Robust Missing Data Handling
# ──────────────────────────────────────────────────────────────

# ── Feature 1 & 6: Exhaustive field synonym catalogue ────────
# Each tuple = (internal_field_name, [all accepted synonyms])
_FIELD_SYNONYMS: Dict[str, List[str]] = {
    "candidate_id": [
        "candidate_id", "id", "ID", "CandidateID", "cand_id", "applicant_id",
        "user_id", "uid", "profile_id",
    ],
    "name": [
        "name", "full_name", "FullName", "anonymized_name", "candidate_name",
        "applicant_name", "Name",
    ],
    "headline": [
        "headline", "Headline", "title", "Title", "job_title", "current_headline",
    ],
    "summary": [
        "summary", "Summary", "about", "bio", "Bio", "description", "Description",
        "profile_summary", "professional_summary", "overview", "objective",
        "career_objective", "about_me",
    ],
    "current_title": [
        "current_title", "CurrentTitle", "role", "Role", "current_role",
        "designation", "Designation", "position", "Position", "job_title",
        "JobTitle", "title", "Title",
    ],
    "current_company": [
        "current_company", "CurrentCompany", "company", "Company", "employer",
        "Employer", "organization", "Organisation", "Organization", "workplace",
        "Workplace", "firm", "Firm", "current_employer",
    ],
    "location": [
        "location", "Location", "city", "City", "current_location",
        "CurrentLocation", "residence", "Residence", "address", "Address",
        "current_city",
    ],
    "years_of_experience": [
        "years_of_experience", "years_experience", "experience",
        "YearsExperience", "yearsExperience", "total_experience", "total_exp",
        "experience_years", "exp", "Exp", "work_experience_years",
        "professional_experience", "years", "experience_in_years",
    ],
    "skills": [
        "skills", "Skills", "skill_set", "skillset", "Skillset",
        "technical_skills", "TechnicalSkills", "technologies", "Technologies",
        "competencies", "Competencies", "expertise", "Expertise",
        "tech_stack", "TechStack", "stack", "Stack",
        "programming_languages", "ProgrammingLanguages", "core_skills",
        "CoreSkills", "tools", "Tools",
    ],
    "career_history": [
        "career_history", "CareerHistory", "work_experience", "WorkExperience",
        "employment_history", "EmploymentHistory", "work_history", "WorkHistory",
        "experience_history", "ExperienceHistory", "professional_history",
        "ProfessionalHistory", "job_history", "positions", "roles",
        "experience_details",
    ],
    "education": [
        "education", "Education", "academics", "Academics",
        "educational_background", "EducationalBackground", "qualification",
        "Qualification", "qualifications", "degrees", "Degrees",
        "academic_background",
    ],
    "certifications": [
        "certifications", "Certifications", "certificates", "Certificates",
        "credentials", "Credentials", "licenses", "Licenses",
        "professional_certifications", "training", "Training",
        "professional_training", "certs",
    ],
    "projects": [
        "projects", "Projects", "portfolio", "Portfolio",
        "relevant_projects", "RelevantProjects", "project_experience",
        "ProjectExperience", "case_studies", "CaseStudies", "assignments",
    ],
    "publications": [
        "publications", "Publications", "papers", "Papers", "research_papers",
        "articles", "Articles",
    ],
    "awards": [
        "awards", "Awards", "achievements", "Achievements", "honors", "Honors",
        "recognition", "Recognition",
    ],
    "open_source": [
        "open_source", "github", "GitHub", "github_url", "github_profile",
        "github_link", "GithubUrl", "GithubProfile", "oss_contributions",
    ],
    "linkedin": [
        "linkedin", "LinkedIn", "linkedin_url", "linkedin_profile",
        "LinkedinUrl", "LinkedinProfile", "linkedin_link",
    ],
    "notice_period_days": [
        "notice_period_days", "notice_period", "NoticePeriod",
        "availability", "Availability", "joining_time", "joining_period",
        "available_in_days", "notice",
    ],
    "github_activity_score": [
        "github_activity_score", "github_score", "GitHubScore",
        "oss_score", "github_stars",
    ],
}

# Pre-build fast lookup: any variant → canonical field name
_VARIANT_TO_CANONICAL: Dict[str, str] = {}
for _canon, _variants in _FIELD_SYNONYMS.items():
    for _v in _variants:
        _VARIANT_TO_CANONICAL[_v] = _canon
        _VARIANT_TO_CANONICAL[_v.lower()] = _canon
        _VARIANT_TO_CANONICAL[_v.upper()] = _canon
        _VARIANT_TO_CANONICAL[_v.title()] = _canon


def _resolve_field(raw: Dict, canonical: str, default=None):
    """Return the first non-empty value from raw matching any synonym of canonical."""
    for variant in _FIELD_SYNONYMS.get(canonical, [canonical]):
        for form in [variant, variant.lower(), variant.upper(), variant.title()]:
            v = raw.get(form)
            if v is not None and str(v).strip() not in ("", "nan", "None", "NaN"):
                return v
    return default


def _resolve_str(raw: Dict, canonical: str, default: str = "") -> str:
    v = _resolve_field(raw, canonical, default)
    return str(v).strip() if v is not None else default


def _resolve_list(raw: Dict, canonical: str) -> List:
    v = _resolve_field(raw, canonical)
    if isinstance(v, list):
        return v
    return []


# ── Feature 3: Experience inference from role-level labels ────
_ROLE_LEVEL_EXP: List[Tuple[List[str], float]] = [
    (["intern", "trainee", "apprentice"],           0.5),
    (["entry", "graduate", "fresher", "fresh grad", "new grad", "entry-level", "entry level"], 1.0),
    (["junior", "jr", "jr."],                       2.0),
    (["associate"],                                  3.0),
    (["mid", "mid-level", "intermediate", "experienced developer"], 4.0),
    (["senior", "sr", "sr."],                        6.0),
    (["lead", "tech lead", "technical lead", "team lead"], 8.0),
    (["principal", "staff"],                        10.0),
    (["architect", "solutions architect", "enterprise architect"], 10.0),
    (["manager", "engineering manager"],            10.0),
    (["director", "head of", "vp", "vice president", "cto", "ceo"], 12.0),
]

def _infer_experience_from_label(text: str) -> Optional[float]:
    """
    Given a freeform experience string like 'Senior' or 'Entry Level',
    return an approximate years value. Returns None if clearly numeric.
    """
    t = text.lower().strip()
    # If it already parses as a number (with units), don't infer
    if re.search(r"\d+", t):
        return None
    for labels, years in _ROLE_LEVEL_EXP:
        if any(lbl in t for lbl in labels):
            return years
    return None

def _parse_years_exp(raw: Dict, current_title: str = "") -> float:
    """
    Feature 3: Parse experience from all synonym keys.
    Supports numeric, range, unit-annotated, and role-label formats.
    Numeric values always take priority over inferred labels.
    """
    # Gather all candidate values across all synonym keys
    for k in _FIELD_SYNONYMS["years_of_experience"]:
        v = raw.get(k)
        if v is None:
            continue
        s = str(v).lower().strip()
        if s in ("", "nan", "none"):
            continue

        # Range: "2-5 years" or "2–5 years" → take midpoint
        range_m = re.match(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", s)
        if range_m:
            lo, hi = float(range_m.group(1)), float(range_m.group(2))
            return round((lo + hi) / 2, 1)

        # "10+" or "10+ years"
        plus_m = re.match(r"(\d+(?:\.\d+)?)\s*\+", s)
        if plus_m:
            return float(plus_m.group(1))

        # Numeric with optional units: "5 years", "6.5 yrs", "3 yr", "4"
        num_m = re.match(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|yr)?$", s)
        if num_m:
            val = float(num_m.group(1))
            if 0 <= val <= 60:
                return val

        # Role-label inference: "Senior", "Fresher", "Entry Level"
        inferred = _infer_experience_from_label(s)
        if inferred is not None:
            return inferred

    # Fallback: infer from current_title if still zero
    if current_title:
        inferred = _infer_experience_from_label(current_title)
        if inferred is not None:
            return inferred

    return 0.0


# ── Feature 2: Semantic Section Detection ─────────────────────
# Patterns that identify section content even without explicit headers
_SECTION_HINTS = {
    "career": [
        r"\b(20\d{2}|19\d{2})\s*[-–]\s*(20\d{2}|present|current)",  # date ranges
        r"\b(senior|lead|principal|junior|associate|engineer|developer|manager|architect)\b.{0,60}\b(at|@|,)\b",
        r"\bworked at\b",
        r"\bjoined\b.{0,40}\b(as|to)\b",
    ],
    "education": [
        r"\b(b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|bca|mca|b\.?s|m\.?s|ph\.?d|mba|diploma)\b",
        r"\b(university|college|institute|school|iit|nit|bits)\b",
        r"\b(graduation|graduated|class of)\b",
    ],
    "skills": [
        r"\b(python|java|javascript|typescript|react|angular|vue|node|django|fastapi|"
        r"pytorch|tensorflow|aws|gcp|azure|docker|kubernetes|sql|mongodb|redis|kafka)\b",
    ],
    "certifications": [
        r"\b(certified|certification|aws certified|google certified|azure certified|"
        r"cka|ckad|pmp|cissp|comptia|microsoft certified)\b",
    ],
    "projects": [
        r"\b(built|developed|designed|implemented|created|launched)\b.{0,80}\b(system|app|platform|tool|service|api|model)\b",
        r"\bgithub\.com/\S+\b",
    ],
}

def _detect_sections_from_text(text: str) -> Dict[str, List[str]]:
    """
    Feature 2: Infer semantic sections from freeform resume text
    when no explicit headings are present. Returns a dict of
    section_name → list of matching snippets (for downstream use).
    """
    detected: Dict[str, List[str]] = {k: [] for k in _SECTION_HINTS}
    lines = text.split("\n")
    for line in lines:
        line_l = line.lower()
        for section, patterns in _SECTION_HINTS.items():
            for pat in patterns:
                if re.search(pat, line_l):
                    detected[section].append(line.strip())
                    break
    return detected


def _extract_skills_from_text(text: str) -> List[Dict]:
    """
    Feature 5: Parse skills from freeform text when no explicit skills field exists.
    Used when semantic section detection identifies skill lines.
    """
    KNOWN_SKILLS = {
        "python", "java", "javascript", "typescript", "go", "golang", "rust", "scala", "kotlin",
        "c", "c++", "c#", "ruby", "php", "swift", "r",
        "react", "angular", "vue", "node", "nodejs", "django", "flask", "fastapi", "spring",
        "pytorch", "tensorflow", "keras", "sklearn", "scikit-learn", "xgboost", "lightgbm",
        "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform", "ansible",
        "sql", "postgres", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "spark", "airflow", "flink", "dbt",
        "llm", "rag", "embeddings", "faiss", "pinecone", "milvus", "weaviate",
        "transformers", "bert", "gpt", "huggingface",
        "mlflow", "wandb", "kubeflow", "sagemaker",
        "git", "linux", "bash", "graphql", "grpc", "rest", "api",
        "nlp", "deep learning", "machine learning", "ai", "computer vision",
        "html", "css", "sass", "webpack", "vite", "next.js", "nuxt",
        "excel", "tableau", "power bi", "looker",
    }
    found = []
    text_l = text.lower()
    for skill in KNOWN_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_l):
            found.append({"name": skill, "proficiency": "intermediate"})
    return found


def _normalize_skills_field(raw_skills) -> List[Dict]:
    """Convert any skills format (list of str, list of dict, CSV string) to unified list."""
    skills = []
    if isinstance(raw_skills, list):
        for s in raw_skills:
            if isinstance(s, dict):
                # Normalize dict: may have 'name'/'skill'/'technology' key
                name = s.get("name") or s.get("skill") or s.get("technology") or s.get("title") or ""
                proficiency = s.get("proficiency") or s.get("level") or s.get("expertise") or "intermediate"
                if name:
                    skills.append({"name": str(name).strip(), "proficiency": str(proficiency).strip()})
            elif isinstance(s, str) and s.strip():
                skills.append({"name": s.strip(), "proficiency": "intermediate"})
    elif isinstance(raw_skills, str) and raw_skills.strip():
        for s in re.split(r"[,;|\n•·▪\-]+", raw_skills):
            s = s.strip()
            if s and len(s) < 60:  # sanity: skip paragraph-length blobs
                skills.append({"name": s, "proficiency": "intermediate"})
    return skills


def _normalize_career_entry(entry) -> Dict:
    """Normalize a single career history entry from any format."""
    if not isinstance(entry, dict):
        return {"title": str(entry), "company": "", "description": "", "duration_months": 0}
    # Accept multiple key names for each sub-field
    title = (entry.get("title") or entry.get("role") or entry.get("designation")
             or entry.get("position") or entry.get("job_title") or "")
    company = (entry.get("company") or entry.get("employer") or entry.get("organization")
               or entry.get("organisation") or entry.get("firm") or "")
    desc = (entry.get("description") or entry.get("responsibilities") or entry.get("summary")
            or entry.get("details") or entry.get("achievements") or "")
    duration = entry.get("duration_months") or entry.get("duration") or 0
    if not isinstance(duration, (int, float)):
        # Try to parse "2 years 3 months" or "27 months"
        dur_s = str(duration).lower()
        months = 0
        yr_m = re.search(r"(\d+)\s*year", dur_s)
        mo_m = re.search(r"(\d+)\s*month", dur_s)
        if yr_m:
            months += int(yr_m.group(1)) * 12
        if mo_m:
            months += int(mo_m.group(1))
        duration = months
    return {
        "title": str(title).strip(),
        "company": str(company).strip(),
        "description": str(desc).strip(),
        "duration_months": int(duration),
    }


def normalize_candidate(raw: Dict) -> Dict:
    """
    Phase 2 — Normalize any candidate format into unified schema.
    Feature 1: Universal field name normalization via _FIELD_SYNONYMS.
    Feature 3: Experience inference from role-level labels.
    Feature 5: Resume format robustness.
    Feature 6: Synonym recognition.
    Feature 7: No fabrication — missing fields get safe defaults, weights redistribute.
    """
    # Fast path: already has a "profile" dict, so we trust profile.* fields
    # (current_title, current_company, etc.) as-is rather than re-deriving them.
    # However, top-level fields like skills/certifications/education can still be
    # under non-canonical synonym names (e.g. "technical_skills" instead of "skills")
    # even when a "profile" dict is present — so we still run synonym resolution
    # for those, rather than returning immediately and silently dropping them.
    if "profile" in raw and isinstance(raw.get("profile"), dict):
        c = raw.copy()
        if "candidate_id" not in c:
            c["candidate_id"] = str(id(raw))
        # Still run career entry normalization on existing profile
        if "career_history" in c and isinstance(c["career_history"], list):
            c["career_history"] = [_normalize_career_entry(e) for e in c["career_history"]]
        else:
            career_raw = _resolve_list(raw, "career_history")
            if career_raw:
                c["career_history"] = [_normalize_career_entry(e) for e in career_raw]

        # Skills — resolve via synonyms if not already a populated top-level "skills" list
        if not c.get("skills"):
            skills_raw = _resolve_field(raw, "skills")
            if skills_raw is not None:
                c["skills"] = _normalize_skills_field(skills_raw)

        # Certifications — resolve via synonyms (list form or delimited string form)
        if not c.get("certifications"):
            certs_raw = _resolve_list(raw, "certifications")
            if not certs_raw:
                cert_raw = _resolve_field(raw, "certifications")
                if isinstance(cert_raw, str) and cert_raw.strip():
                    certs_raw = [{"name": cs.strip()} for cs in re.split(r"[,;|\n]", cert_raw) if cs.strip()]
            if certs_raw:
                c["certifications"] = certs_raw

        # Education / projects / publications / awards — same synonym fallback
        for field in ("education", "projects", "publications", "awards"):
            if not c.get(field):
                resolved = _resolve_list(raw, field)
                if resolved:
                    c[field] = resolved

        # Open-source / LinkedIn links
        if not c.get("open_source"):
            os_val = _resolve_str(raw, "open_source")
            if os_val:
                c["open_source"] = os_val
        if not c.get("linkedin"):
            li_val = _resolve_str(raw, "linkedin")
            if li_val:
                c["linkedin"] = li_val

        return c

    # ── Identity ──────────────────────────────────────────────
    candidate_id = _resolve_str(raw, "candidate_id") or f"CAND_{abs(id(raw))}"
    name = _resolve_str(raw, "name") or "Unknown"

    # ── Title / Headline ─────────────────────────────────────
    headline = _resolve_str(raw, "headline")
    current_title = _resolve_str(raw, "current_title") or headline

    # ── Company ──────────────────────────────────────────────
    current_company = _resolve_str(raw, "current_company")

    # ── Location ─────────────────────────────────────────────
    location = _resolve_str(raw, "location")

    # ── Summary / Bio ────────────────────────────────────────
    summary = _resolve_str(raw, "summary")

    # ── Feature 3: Experience (numeric → range → label → title inference) ──
    years_exp = _parse_years_exp(raw, current_title)

    # ── Feature 1 & 6: Skills (all synonym keys) ─────────────
    skills_raw = _resolve_field(raw, "skills")
    skills = _normalize_skills_field(skills_raw) if skills_raw is not None else []

    # ── Feature 1 & 6: Career history (all synonym keys) ─────
    career_raw = _resolve_list(raw, "career_history")
    career = [_normalize_career_entry(e) for e in career_raw] if career_raw else []

    # ── Feature 2: Semantic section detection fallback ────────
    # If no structured career/skills found, try to detect from freeform text
    raw_text_blob = summary
    if not raw_text_blob:
        # Stitch together all string values for section detection
        raw_text_blob = " \n".join(
            str(v) for v in raw.values()
            if isinstance(v, str) and len(str(v)) > 20
        )

    if not career and raw_text_blob:
        detected = _detect_sections_from_text(raw_text_blob)
        if detected.get("career"):
            # Build a synthetic career entry from detected lines
            career = [{
                "title": current_title,
                "company": current_company,
                "description": "\n".join(detected["career"]),
                "duration_months": int(years_exp * 12),
            }]

    if not skills and raw_text_blob:
        detected_skills = _extract_skills_from_text(raw_text_blob)
        if detected_skills:
            skills = detected_skills

    # If still no career, synthesize from available fields
    if not career and (current_title or current_company or summary):
        career = [{
            "title": current_title,
            "company": current_company,
            "description": summary,
            "duration_months": int(years_exp * 12),
        }]

    # ── Feature 1: Education (all synonym keys) ───────────────
    education = _resolve_list(raw, "education")

    # ── Feature 1: Certifications (all synonym keys) ──────────
    certifications = _resolve_list(raw, "certifications")
    # Also accept string lists of cert names
    if not certifications:
        cert_raw = _resolve_field(raw, "certifications")
        if isinstance(cert_raw, str) and cert_raw.strip():
            certifications = [{"name": c.strip()} for c in re.split(r"[,;|\n]", cert_raw) if c.strip()]

    # ── Feature 1: Projects (all synonym keys) ────────────────
    projects = _resolve_list(raw, "projects")

    # ── Feature 1: OSS / GitHub / LinkedIn ───────────────────
    open_source = _resolve_str(raw, "open_source")
    linkedin = _resolve_str(raw, "linkedin")

    # ── Feature 1: Notice period (all synonym keys) ──────────
    notice_raw = _resolve_field(raw, "notice_period_days")
    notice_days = 60  # default
    if notice_raw is not None:
        ns = str(notice_raw).lower().strip()
        # "immediate" / "0" → 0
        if ns in ("0", "0.0") or any(w in ns for w in ["immediate", "asap"]):
            notice_days = 0
        else:
            # Convert weeks/months → days
            val_m = re.search(r"(\d+)", ns)
            if val_m:
                val = int(val_m.group(1))
                if "month" in ns:
                    val *= 30
                elif "week" in ns:
                    val *= 7
                notice_days = val

    # ── Feature 1: GitHub activity ────────────────────────────
    gh_score_raw = _resolve_field(raw, "github_activity_score")
    gh_score = -1
    if gh_score_raw is not None:
        try:
            gh_score = float(gh_score_raw)
        except (ValueError, TypeError):
            pass

    # ── Redrob signals (structured if present, else defaults) ─
    redrob = raw.get("redrob_signals", {})
    if not isinstance(redrob, dict):
        redrob = {}

    publications = _resolve_list(raw, "publications")
    awards = _resolve_list(raw, "awards")

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
            "current_company_size": _resolve_str(raw, "current_company_size"),
            "current_industry": (
                _resolve_str(raw, "current_industry")
                or str(raw.get("industry", raw.get("Industry", "")))
            ),
        },
        "career_history": career,
        "education": education,
        "skills": skills,
        "certifications": certifications,
        "projects": projects,
        "publications": publications,
        "awards": awards,
        "open_source": open_source or str(redrob.get("github_url", "")),
        "linkedin": linkedin,
        "redrob_signals": {
            "last_active_date": redrob.get("last_active_date", "2024-06-01"),
            "recruiter_response_rate": float(redrob.get("recruiter_response_rate", 0.7)),
            "interview_completion_rate": float(redrob.get("interview_completion_rate", 0.7)),
            "open_to_work_flag": bool(redrob.get("open_to_work_flag", True)),
            "github_activity_score": gh_score if gh_score > 0 else float(redrob.get("github_activity_score", -1)),
            "notice_period_days": redrob.get("notice_period_days", notice_days),
            "profile_completeness_score": float(redrob.get("profile_completeness_score", 70)),
        },
        "_raw_text": summary or raw_text_blob[:2000],
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
        # date.fromisoformat avoids the heavy locale checks of strptime
        return (TODAY - date.fromisoformat(str(date_str)[:10])).days
    except:
        return 999

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

# Cache compiled \b<word>\b patterns — these are re-used across every candidate in a
# ranking run (same JD skill list, same nice-to-have list), so compiling them once per
# unique word instead of once per (candidate x skill) call removes the dominant cost in
# skill_alignment_score. Unbounded but keyed on short skill/keyword strings only, so the
# cache stays small relative to candidate pool size.
_WORD_BOUNDARY_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}
_JD_KEYWORD_TERM_CACHE: Dict[str, set] = {}

def _get_word_boundary_pattern(word: str) -> "re.Pattern":
    pat = _WORD_BOUNDARY_PATTERN_CACHE.get(word)
    if pat is None:
        pat = re.compile(r'\b' + re.escape(word) + r'\b')
        _WORD_BOUNDARY_PATTERN_CACHE[word] = pat
    return pat

def _skill_match(skill_l: str, skill_names: set, text_lower: str) -> float:
    canon = _normalize_skill(skill_l)
    normalized_names = {_normalize_skill(sn) for sn in skill_names}
    if canon in normalized_names:
        return 1.0
    if len(canon) > 3 and canon not in _WHOLE_WORD_ONLY:
        for sn in normalized_names:
            if (canon in sn or sn in canon) and len(min(canon, sn, key=len)) > 3:
                return 1.0
    if _get_word_boundary_pattern(canon).search(text_lower):
        return 0.6
    if canon != skill_l and _get_word_boundary_pattern(skill_l).search(text_lower):
        return 0.6
    return 0.0

_GENERIC_JD_WORDS = {
    "the", "and", "for", "you", "are", "with", "this", "that", "your", "our", "will",
    "have", "has", "from", "team", "work", "role", "job", "years", "year", "experience",
    "skills", "ability", "strong", "good", "excellent", "looking", "candidate", "candidates",
    "should", "must", "ideal", "preferred", "responsibilities", "requirements", "about",
    "company", "join", "etc", "such", "into", "their", "they", "them", "well", "than",
    "also", "use", "using", "used", "all", "can", "who", "what", "when", "where", "how",
}

def skill_alignment_score(candidate_skills: List, must_have: List[str], nice_to_have: List[str], full_text: str, jd_text: str = "") -> Tuple[float, List[str]]:
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

    # JD keyword bonus — reward genuine overlap between *the job description's own
    # wording* and the candidate's text, beyond the explicit must/nice-to-have lists.
    # (Previously this compared the candidate's own skill list against their own bio
    # text, which is close to a tautology and inflated almost every candidate equally.)
    # `extra_terms` depends only on jd_text/must_have/nice_to_have, which are identical
    # for every candidate in a single ranking run — cache the term set keyed by jd_text
    # instead of recomputing it per candidate. The match itself is done by tokenizing the
    # candidate's text once into a word set and intersecting, rather than running one
    # regex search per JD term per candidate (O(terms) regex scans -> O(text length)
    # tokenization + O(1) set lookups). For a long narrative JD with 500+ extra terms
    # this is the difference between tens of millions of regex scans and a single pass.
    jd_kw_bonus = 0.0
    if jd_text:
        cache_key = jd_text
        extra_terms_set = _JD_KEYWORD_TERM_CACHE.get(cache_key)
        if extra_terms_set is None:
            already_covered = {s.lower().strip() for s in must_have + nice_to_have}
            jd_words = set(re.findall(r"[a-z][a-z0-9+#.]{3,}", jd_text.lower()))
            extra_terms_set = {w for w in jd_words if w not in already_covered and w not in _GENERIC_JD_WORDS}
            _JD_KEYWORD_TERM_CACHE[cache_key] = extra_terms_set
        if extra_terms_set:
            candidate_words = set(re.findall(r"[a-z][a-z0-9+#.]{3,}", text_lower))
            jd_kw_bonus = 0.05 * len(extra_terms_set & candidate_words)
    jd_kw_bonus = min(jd_kw_bonus, 1.0)

    max_possible = max(1.0, len(must_have) * 3.0 + len(nice_to_have) * 1.0 + 1.5)
    return min(1.0, (score + jd_kw_bonus) / max_possible), matched[:12]

_PRODUCTION_STRONG_POS_PATTERNS = [re.compile(p) for p in [
    r"shipped", r"production", r"deployed", r"deployment", r"scaling", r"scaled",
    r"a/b test", r"launched", r"released", r"migrated", r"real-time", r"realtime",
    r"\d+m\s+users", r"million users", r"billion", r"latency", r"monitoring",
    r"serving", r"inference at scale", r"reduced .{0,30}by \d+", r"improved .{0,30}by \d+",
    r"optimized", r"processing \d+",
]]
_PRODUCTION_WEAK_POS = ["built", "implemented", "integrated", "automated", "pipeline",
                         "feature", "model in production", "end-to-end", "end to end"]
_PRODUCTION_NEG = ["research only", "no production", "theoretical", "academic only", "arxiv", "published paper"]

def production_signal_score(text: str) -> float:
    text_l = text.lower()
    pos_strong = sum(1 for p in _PRODUCTION_STRONG_POS_PATTERNS if p.search(text_l))
    pos_weak   = sum(1 for p in _PRODUCTION_WEAK_POS if p in text_l)
    neg        = sum(1 for n in _PRODUCTION_NEG      if n in text_l)

    net = pos_strong * 2.0 + pos_weak * 0.5 - neg * 2.0
    if net <= 0 and pos_strong == 0 and pos_weak == 0:
        return 0.45
    return max(0.0, min(1.0, net / 12.0))

def certification_score(certifications: List, mandatory_certs: List[str], preferred_certs: List[str], full_text: str) -> float:
    """
    Score certification alignment against JD requirements.
    Returns 0-1. Falls back to a neutral score when the JD names no specific
    certifications, but still gives modest credit for having any at all.
    """
    cert_names = set()
    for cert in certifications:
        raw = cert.get("name", "") if isinstance(cert, dict) else str(cert)
        raw = raw.lower().strip()
        if raw:
            cert_names.add(raw)

    text_lower = full_text.lower()

    if not mandatory_certs and not preferred_certs:
        # JD didn't name specific certifications — reward having verifiable ones at all
        return min(1.0, 0.5 + 0.15 * len(cert_names)) if cert_names else 0.5

    score, max_possible = 0.0, 0.0
    for cert in mandatory_certs:
        cl = cert.lower().strip()
        if not cl:
            continue
        max_possible += 2.0
        if any(cl in cn or cn in cl for cn in cert_names) or _get_word_boundary_pattern(cl).search(text_lower):
            score += 2.0
    for cert in preferred_certs:
        cl = cert.lower().strip()
        if not cl:
            continue
        max_possible += 1.0
        if any(cl in cn or cn in cl for cn in cert_names) or _get_word_boundary_pattern(cl).search(text_lower):
            score += 1.0

    if max_possible == 0:
        return 0.5
    return min(1.0, score / max_possible)

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
    known_startupish = {"razorpay", "swiggy", "zomato", "meesho", "zepto", "cred", "phonepe",
                         "browserstack", "freshworks"}
    if any(sc in c for sc in service_cos): return 0.3 if prefers_product else 0.6
    if prefers_startup and any(sk in c for sk in known_startupish): return 1.0
    if any(pc in c for pc in known_product): return 0.8 if prefers_startup else 1.0
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
    if bd.get("certifications", 0) >= 0.75: strengths.append("Certifications aligned with role requirements")
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
def _heuristic_semantic_score(jd_tokens_set, full_text):
    """
    Fast token-Jaccard fallback (no model calls). Shared by the
    no-transformers path and by the adaptive deadline cutover in
    precompute_semantic_scores() below, so both code paths compute
    the exact same fallback score the exact same way.
    """
    cand_tokens_set = set(re.findall(r"[a-z][a-z0-9]{2,}", full_text.lower()))
    if jd_tokens_set or cand_tokens_set:
        intersection = len(jd_tokens_set & cand_tokens_set)
        union = len(jd_tokens_set | cand_tokens_set)
        return min(1.0, (intersection / union if union else 0.0) * 4.0)
    return 0.0


def precompute_semantic_scores(
    jd_text: str,
    candidates: List[Dict],
    batch_size: int = 256,
    deadline: float = None,
) -> List[float]:
    """
    Batch-compute semantic_s (bi-encoder cosine + cross-encoder relevance) for
    ALL candidates in one pass, instead of letting score_candidate() re-encode
    the JD and call the models one candidate at a time.

    `deadline` (optional): a time.time()-style epoch timestamp. If, after
    timing a batch, another batch of the same size would plausibly finish
    after this deadline, we stop calling the models and fill every
    remaining candidate with the fast token-Jaccard heuristic instead.
    This guarantees the function returns within budget on any hardware,
    regardless of how slow bi-encoder/cross-encoder inference turns out
    to be on the machine it's running on, instead of silently blowing
    the 5-minute compute constraint.

    Falls back to the existing token-Jaccard heuristic per candidate when
    sentence-transformers isn't installed.

    Returns a list of semantic_s floats aligned 1:1 with `candidates`.
    """
    full_texts = [
        build_candidate_text(apply_fairness_filter(c)) for c in candidates
    ]
    n = len(full_texts)
    jd_tokens_set = set(re.findall(r"[a-z][a-z0-9]{2,}", jd_text.lower()))

    if not HAS_TRANSFORMERS:
        return [_heuristic_semantic_score(jd_tokens_set, t) for t in full_texts]

    import time as _time
    jd_emb = bi_encoder.encode(jd_text[:2000], convert_to_tensor=True)

    cos_sims = [0.0] * n
    cross_norms = [0.0] * n
    used_heuristic = [False] * n

    n_batches = (n + batch_size - 1) // batch_size
    cutover_at_batch = None
    for batch_num, start in enumerate(range(0, n, batch_size), 1):
        end = min(start + batch_size, n)

        if deadline is not None and _time.time() >= deadline:
            cutover_at_batch = batch_num
            break

        batch_texts = [t[:2000] for t in full_texts[start:end]]
        t_b0 = _time.time()

        cand_embs = bi_encoder.encode(
            batch_texts, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=False
        )
        batch_cos = F.cosine_similarity(jd_emb.unsqueeze(0), cand_embs, dim=1)
        for i, v in enumerate(batch_cos.tolist()):
            cos_sims[start + i] = v
        t_b1 = _time.time()

        pair_batch = [[jd_text[:1000], t[:1000]] for t in full_texts[start:end]]
        cross_raw_batch = cross_encoder.predict(pair_batch, batch_size=batch_size, show_progress_bar=False)
        for i, raw in enumerate(cross_raw_batch):
            cross_norms[start + i] = float(1 / (1 + math.exp(-raw)))
        t_b2 = _time.time()

        print(f"      batch {batch_num}/{n_batches} ({end}/{n}) -- "
              f"bi-encoder {t_b1 - t_b0:.1f}s, cross-encoder {t_b2 - t_b1:.1f}s, "
              f"total {t_b2 - t_b0:.1f}s", flush=True)

        if deadline is not None:
            batch_cost = t_b2 - t_b0
            if _time.time() + batch_cost > deadline:
                cutover_at_batch = batch_num + 1
                break

    if cutover_at_batch is not None:
        cutover_start = (cutover_at_batch - 1) * batch_size
        remaining = n - cutover_start
        print(f"      [BUDGET] deadline reached after batch {cutover_at_batch - 1}/{n_batches} -- "
              f"falling back to fast heuristic scoring for the remaining "
              f"{remaining} candidates ({cutover_start}..{n}) to stay within budget.", flush=True)
        for i in range(cutover_start, n):
            cos_sims[i] = _heuristic_semantic_score(jd_tokens_set, full_texts[i])
            cross_norms[i] = cos_sims[i]
            used_heuristic[i] = True

    return [
        cos_sims[i] if used_heuristic[i] else (0.40 * cos_sims[i] + 0.60 * cross_norms[i])
        for i in range(n)
    ]



def score_candidate(c, bm25_norm, bm25_max, jd_analysis, jd_text, jd_weights=None, semantic_score=None):
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
    full_text = c.get('cached_text', build_candidate_text(c))

    # ── Generate per-JD weights (once, cached by caller ideally) ──
    if jd_weights is None:
        jd_weights = generate_jd_weights(jd_analysis, jd_text)

    # ═══════════════════════════════════════════════════════════
    # COMPUTE ALL RAW SIGNALS
    # ═══════════════════════════════════════════════════════════

    # 1. Skill alignment
    skill_s, matched_skills = skill_alignment_score(
        skills, jd_analysis.get("must_have_skills", []),
        jd_analysis.get("nice_to_have_skills", []), full_text, jd_text,
    )

    # 2. Experience fit
    years   = float(profile.get("years_of_experience", 0))
    exp_s   = exp_score(years, jd_analysis.get("years_min", 2), jd_analysis.get("years_max", 10))

    # 3. Career progression
    career_s, career_detail = analyze_career_progression(career)
    has_career = bool(career)

    # 4. Semantic similarity
    if semantic_score is not None:
        # Precomputed in batch by precompute_semantic_scores() -- avoids
        # re-encoding the JD and calling the cross-encoder unbatched,
        # once per candidate (the old path below does not scale to 5000+ calls).
        semantic_s = semantic_score
    elif HAS_TRANSFORMERS:
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

    # 13. Certifications (was previously weighted but never scored — now wired in)
    cert_s = certification_score(
        c_filtered.get("certifications", []),
        jd_analysis.get("mandatory_certifications", []),
        jd_analysis.get("preferred_certifications", []),
        full_text,
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
    tech_total_w = w["skill_alignment"] + w["semantic_match"] + w["production_signals"] + w["bm25_match"] + w["github_oss"] + w["certifications"]
    if tech_total_w > 0:
        technical_fit = (
            w["skill_alignment"]    * skill_s
            + w["semantic_match"]   * semantic_s
            + w["production_signals"] * ship_s
            + w["bm25_match"]       * bm25_s
            + w["github_oss"]       * github_s
            + w["certifications"]   * cert_s
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

    # Education as multiplicative modifier (PhD for PhD role → small uplift).
    # Only apply when we actually found education data to compare — otherwise this
    # silently penalized every candidate with no parseable education info (edu_mod
    # defaults to 0.95), which contradicts the "never penalize missing data" design.
    edu_multiplier = edu_mod if has_edu else 1.0
    base = max(0.0, min(1.0, base * edu_multiplier))

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
        "certifications":     round(cert_s,         4),
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

    # Step 1: Analyze JD (Heuristic parser is the explicit primary design)
    jd_analysis = extract_jd_heuristic(jd_text)

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

    scored.sort(key=lambda x: (-x[0], x[1]['candidate_id']))
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
    # ── JD resolution: file → OCR/parse → fallback to text box ──
    # Priority: file parse result if non-empty, then text-box input.
    # This handles: (a) image file with OCR, (b) file parse fails/empty → text box,
    # (c) text box only, (d) both provided where file is empty (image + no OCR installed).
    jd_content = ""

    if jd_file and jd_file.filename:
        jd_bytes = await jd_file.read()
        jd_content = parse_jd_file(jd_bytes, jd_file.filename)
        # If file parse returned empty (e.g. image without OCR, or corrupt file),
        # fall through to text-box input below rather than immediately erroring.

    if not jd_content.strip() and jd_text and jd_text.strip():
        jd_content = jd_text.strip()

    if not jd_content.strip():
        if jd_file and jd_file.filename:
            ext = jd_file.filename.lower().rsplit(".", 1)[-1]
            if ext in ("jpg", "jpeg", "png"):
                raise HTTPException(
                    400,
                    "Image job descriptions cannot be read without an OCR engine. "
                    "Please paste the job description text into the text box instead."
                )
            if ext == "pdf":
                raise HTTPException(
                    400,
                    "Could not extract text from the PDF (it may be a scanned/image-only PDF). "
                    "Please paste the job description text into the text box instead."
                )
        raise HTTPException(400, "Job description is empty — provide a file or paste text")

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
        "title_relevance_%", "engagement_%", "github_oss_%", "certifications_%",
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
            pct(bd.get("github_oss")), pct(bd.get("certifications")),
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