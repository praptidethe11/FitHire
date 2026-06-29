#!/usr/bin/env python3
"""
FitHire CLI — for bulk processing challenge datasets
=========================================================
Usage:
    python -m backend.rank_cli \
        --candidates candidates.jsonl \
        --jd job_description.txt \
        --out submission.csv \
        --top 100
"""

import argparse
import asyncio
import csv
import json
import re
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import (
    normalize_candidate,
    analyze_jd_with_claude,
    extract_jd_heuristic,
    tokenize_for_bm25,
    normalize_bm25_scores,
    score_candidate,
    build_reasoning,
    build_candidate_text,
    generate_jd_weights,
)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("[WARN] rank_bm25 not installed — BM25 scoring disabled")


def load_jd(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def stream_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except:
                    pass


def load_candidates(path: str):
    if path.endswith(".jsonl"):
        return list(stream_jsonl(path))
    elif path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    elif path.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(path).to_dict(orient="records")
    else:
        return list(stream_jsonl(path))


async def run_cli(candidates_path: str, jd_path: str, out_path: str, top_n: int, use_ai: bool):
    t_start = time.time()

    print("=" * 55)
    print("  FitHire CLI — Candidate Ranking Pipeline")
    print("=" * 55)

    # Load JD
    print(f"\n[1] Loading JD from: {jd_path}")
    jd_text = load_jd(jd_path)
    print(f"    JD length: {len(jd_text)} chars")

    # Analyze JD
    if use_ai:
        print("[2] AI analysis of JD (Claude)...")
        jd_analysis = await analyze_jd_with_claude(jd_text)
    else:
        print("[2] Heuristic JD analysis...")
        jd_analysis = extract_jd_heuristic(jd_text)

    print(f"    Role: {jd_analysis.get('role_title')}")
    print(f"    Seniority: {jd_analysis.get('seniority')}")
    print(f"    Experience: {jd_analysis.get('years_min')}-{jd_analysis.get('years_max')} yrs")
    print(f"    Must-have skills: {', '.join(jd_analysis.get('must_have_skills', [])[:8])}")

    # Generate adaptive weights based on the Job Description
    jd_weights = generate_jd_weights(jd_analysis, jd_text)

    print(f"    Role type detected: {jd_weights.get('_role_type', 'generic')}")

    # JD tokens
    jd_tokens = re.findall(r"[a-z0-9]+", jd_text.lower())
    for kw in jd_analysis.get("domain_keywords", []):
        jd_tokens.extend(re.findall(r"[a-z0-9]+", kw.lower()))

    # Load candidates
    print(f"\n[3] Loading candidates from: {candidates_path}")
    raw = load_candidates(candidates_path)
    print(f"    Total: {len(raw):,} candidates")

    # Normalize
    print("[4] Normalizing candidate data...")
    candidates = [normalize_candidate(c) for c in raw]

    # BM25 index
    print("[5] Building BM25 index...")
    if HAS_BM25:
        corpus = [tokenize_for_bm25(build_candidate_text(c)) for c in candidates]
        bm25_model = BM25Okapi(corpus)
        bm25_raw = list(bm25_model.get_scores(jd_tokens))
        bm25_scores = normalize_bm25_scores(bm25_raw)  # percentile-based, not max-based
    else:
        bm25_scores = [0.0] * len(candidates)

    t_bm25 = time.time()
    print(f"    BM25 index built in {t_bm25 - t_start:.1f}s")

    # Score
    print("[6] Scoring candidates...")
    scored = []

    for i, c in enumerate(candidates):
        # bm25_scores[i] is already [0,1]; pass 1.0 as bm25_max (unused in new scorer)
        final, info = score_candidate(c,bm25_scores[i],1.0,jd_analysis,jd_text,jd_weights)
        scored.append((final, info))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    t_score = time.time()
    print(f"    Scored in {t_score - t_bm25:.1f}s")
    top_score = top[0][1]['final_score'] * 100
    bot_score = top[-1][1]['final_score'] * 100
    print(f"    Score range: {bot_score:.1f}% – {top_score:.1f}%")

    # Write output
    print(f"\n[7] Writing {len(top)} results to: {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id",
            "rank",
            "score_%",
            "confidence_%",
            "reasoning",
            "name",
            "title",
            "company",
            "years_exp",
            "location",
            "skills_matched",
            "skill_alignment_%",
            "experience_fit_%",
            "career_progression_%",
            "notice_period_%",
            "semantic_match_%",
            "technical_fit_%",
            "career_fit_%",
            "recruiter_fit_%",
        ])
        for rank, (_, info) in enumerate(top, 1):
            reasoning = build_reasoning(info, jd_analysis)
            bd = info.get("breakdown", {})
            def pct(v): return f"{round(v * 100, 1)}" if v is not None else ""
            writer.writerow([
                info["candidate_id"],
                rank,
                pct(info["final_score"]),
                pct(info["confidence_score"]),
                reasoning,
                info.get("name", ""),
                info.get("current_title", ""),
                info.get("current_company", ""),
                info.get("years_exp", ""),
                info.get("location", ""),
                "; ".join(info.get("matched_skills", [])),
                pct(bd.get("skill_alignment")),
                pct(bd.get("experience_fit")),
                pct(bd.get("career_progression")),
                pct(bd.get("notice_period")),
                pct(bd.get("semantic_match")),
                pct(bd.get("technical_fit")),
                pct(bd.get("career_fit")),
                pct(bd.get("recruiter_fit")),
            ])

    t_end = time.time()
    print(f"\n✅ Done in {t_end - t_start:.1f}s total")
    print(f"\nTop 10:")
    for rank, (_, info) in enumerate(top[:10], 1):
        score_pct = info['final_score'] * 100
        print(f"  #{rank:2d}  {info['candidate_id']:15s}  {score_pct:5.1f}%  {info['current_title'][:30]}")


def main():
    parser = argparse.ArgumentParser(description="FitHire CLI — Candidate Ranking")
    parser.add_argument("--candidates", required=True, help="Path to candidates file (JSONL/JSON/CSV)")
    parser.add_argument("--jd", required=True, help="Path to job description (TXT/DOCX/PDF)")
    parser.add_argument("--out", default="ranked_candidates.csv", help="Output CSV path")
    parser.add_argument("--top", type=int, default=100, help="Number of candidates to shortlist")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI JD analysis (use heuristics)")
    args = parser.parse_args()

    asyncio.run(run_cli(
        args.candidates,
        args.jd,
        args.out,
        args.top,
        use_ai=not args.no_ai,
    ))


if __name__ == "__main__":
    main()