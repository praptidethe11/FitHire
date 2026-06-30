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


def load_candidates_streaming(path: str):
    """
    Yield candidates one at a time instead of materializing the full list.
    For .jsonl this avoids ever holding all raw records in memory at once —
    each record is normalized and discarded by the caller as it's consumed.
    .json/.csv formats are not naturally streamable (they require a full
    parse), so they fall back to load_candidates() and are yielded from
    the resulting list; the caller still benefits because the *raw* list
    is dropped as soon as iteration finishes (no separate raw+normalized
    list held simultaneously by this function).
    """
    if path.endswith(".jsonl"):
        yield from stream_jsonl(path)
    else:
        for c in load_candidates(path):
            yield c


async def run_cli(candidates_path: str, jd_path: str, out_path: str, top_n: int, use_ai: bool):
    t_start = time.time()

    print("=" * 55)
    print("  FitHire CLI — Candidate Ranking Pipeline")
    print("=" * 55)

    # Load JD
    print(f"\n[1] Loading JD from: {jd_path}")
    jd_text = load_jd(jd_path)
    print(f"    JD length: {len(jd_text)} chars")

    # Analyze JD (Heuristic parser is the explicit primary design)
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

    # Load candidates — stream from disk rather than materializing the full
    # raw list, since raw JSONL records (career_history, descriptions, etc.)
    # are large and we must not hold raw + normalized + corpus copies of all
    # 100K candidates in memory at once (this previously OOM'd at ~3.9GB on a
    # 100K-row dataset).
    print(f"\n[3] Streaming candidates from: {candidates_path}")

    # Pass 1: normalize + build BM25 corpus tokens, freeing each raw record
    # as we go. Only the lean BM25 corpus (token lists) and a same-length
    # list of normalized candidates are kept — not the raw JSONL dicts.
    print("[4] Normalizing candidate data + building BM25 corpus (streaming)...")
    candidates = []
    corpus = []
    n_total = 0
    for raw_c in load_candidates_streaming(candidates_path):
        n_total += 1
        c = normalize_candidate(raw_c)
        c['cached_text'] = build_candidate_text(c)
        candidates.append(c)
        if HAS_BM25:
            corpus.append(tokenize_for_bm25(build_candidate_text(c)))
        if n_total % 20000 == 0:
            print(f"    ...{n_total:,} candidates processed")
    print(f"    Total: {n_total:,} candidates")

    # BM25 index
    print("[5] Building BM25 index...")
    if HAS_BM25:
        valid_pairs = [(i, doc) for i, doc in enumerate(corpus) if doc]
        if valid_pairs:
            valid_indices, valid_docs = zip(*valid_pairs)
            bm25_model = BM25Okapi(valid_docs)
            raw_scores_valid = list(bm25_model.get_scores(jd_tokens))
            bm25_raw = [0.0] * len(candidates)
            for idx, score in zip(valid_indices, raw_scores_valid):
                bm25_raw[idx] = float(score)
            bm25_scores = normalize_bm25_scores(bm25_raw)  # percentile-based, not max-based
        else:
            bm25_scores = [0.0] * len(candidates)
        # corpus (token lists) and the BM25 model are no longer needed past
        # this point — drop references so they can be freed before scoring.
        del corpus
        if 'bm25_model' in dir():
            del bm25_model
    else:
        bm25_scores = [0.0] * len(candidates)

    t_bm25 = time.time()
    print(f"    BM25 index built in {t_bm25 - t_start:.1f}s")

    # Score — keep only (final_score, info) per candidate going forward;
    # `candidates` (normalized dicts incl. full text) is dropped right after
    # scoring since `info` already carries everything build_reasoning needs.
    print("[6] Scoring candidates...")
# 1. Filter: Identify Top 5,000 candidates by BM25 score
    # We use a list of tuples to keep track of original indices while sorting
    indexed_scores = list(enumerate(bm25_scores))
    # Sort by score descending and take the top 5000
    top_indices = [i for i, score in sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:5000]]
    
    scored = []
    
    # 2. Score the Top 5,000 using full AI pipeline
    print(f"    Scoring top 5,000 candidates via AI...")
    for idx in top_indices:
        c = candidates[idx]
        final, info = score_candidate(c, bm25_scores[idx], 1.0, jd_analysis, jd_text, jd_weights=jd_weights)
        scored.append((info["final_score"], info))
        
    # 3. For the remaining 95,000, assign a baseline score of 0.1
    # This keeps the submission valid without wasting compute on weak matches
    print(f"    Assigning baseline to remaining candidates...")
    all_indices = set(range(len(candidates)))
    remaining_indices = all_indices - set(top_indices)
    
    for idx in remaining_indices:
        # Create minimal valid 'info' dict for the CSV output
        c = candidates[idx]
        info = {
            "candidate_id": c["candidate_id"],
            "final_score": 0.1,
            "name": c.get("profile", {}).get("anonymized_name", "Unknown"),
            "breakdown": {"technical_fit": 0.1, "career_fit": 0.1, "recruiter_fit": 0.1}
        }
        scored.append((0.1, info))
    del candidates, bm25_scores

    # Tiebreak per spec Section 3: "If two candidates have the same score,
    # you must still assign unique ranks... by candidate_id ascending."
    # Mirrors backend/main.py's run_ranking_pipeline sort key.
    scored.sort(key=lambda x: (-round(x[0], 7), x[1]["candidate_id"]))
    top = scored[:top_n]

    t_score = time.time()
    print(f"    Scored in {t_score - t_bm25:.1f}s")
    top_score = top[0][1]['final_score'] * 100
    bot_score = top[-1][1]['final_score'] * 100
    print(f"    Score range: {bot_score:.1f}% – {top_score:.1f}%")

    # Write output — EXACT schema required by submission_spec.md Section 2:
    # candidate_id,rank,score,reasoning (4 columns, this order, header verbatim).
    # score is the raw [0,1] final_score, not a "_%"-suffixed percentage —
    # the spec only requires non-increasing-by-rank, not any particular scale.
    print(f"\n[7] Writing {len(top)} results to: {out_path}")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (_, info) in enumerate(top, 1):
            reasoning = build_reasoning(info, jd_analysis)
            writer.writerow([
                info["candidate_id"],
                rank,
                round(info["final_score"], 7),
                reasoning,
            ])

    t_end = time.time()
    print(f"\n[INFO] Done in {t_end - t_start:.1f}s total")
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
    # Compliance note: the ranking step (this CLI) must not call hosted LLM APIs per the
    # hackathon's compute constraints. Heuristic JD analysis is therefore the DEFAULT, not
    # an opt-out — running this command with no flags, exactly as documented in the README,
    # can never trigger a hosted API call even if an API key happens to be present in the
    # environment. --use-ai is an explicit opt-in for local/dev exploration only and must
    # not be used for the submission run.
    parser.add_argument("--use-ai", action="store_true",
                         help="Opt in to hosted LLM JD analysis (DEV/EXPLORATION ONLY — "
                              "not permitted for the actual submission run; the ranking "
                              "step must not call hosted LLM APIs per the compute constraints)")
    parser.add_argument("--no-ai", action="store_true",
                         help="Deprecated, kept for backward compatibility — heuristic "
                              "analysis is now the default regardless of this flag")
    args = parser.parse_args()

    if args.use_ai:
        print("[WARN] --use-ai requests a hosted LLM call for JD analysis. This is NOT "
              "permitted for the competition ranking step (submission_spec.md, Section 3: "
              "'You CANNOT, during the ranking step: Call hosted LLM APIs.'). Use this flag "
              "only for local exploration, never to produce your submission CSV.\n")

    asyncio.run(run_cli(
        args.candidates,
        args.jd,
        args.out,
        args.top,
        use_ai=args.use_ai,
    ))


if __name__ == "__main__":
    main()