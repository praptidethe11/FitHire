#!/usr/bin/env python3
"""
FitHire — post-run submission finalizer
========================================
Run this AFTER rank_cli.py has produced your ranking CSV. It:
  1. Validates the CSV against the hard format rules in submission_spec.md
     Section 3 (exactly 100 rows, ranks 1..100 each used once, unique
     candidate_ids, non-increasing score) so you catch a Section-6
     "common rejection" locally instead of on the portal.
  2. Copies (never moves/renames-in-place) the validated file to
     <participant_id>.csv, the exact filename the spec requires.

Usage:
    python finalize_submission.py --in submission.csv --participant-id team_xxx

This does NOT touch rank_cli.py or how the CSV is produced — it only
validates and mirrors the already-produced file, per the "don't touch the
core ranking logic" constraint.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path


def validate(path: Path) -> list[str]:
    """Returns a list of problems found (empty list = clean)."""
    problems = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return ["File is empty — no header row."]

        expected_header = ["candidate_id", "rank", "score", "reasoning"]
        if header[:4] != expected_header:
            problems.append(
                f"Header must be exactly {expected_header} (in this order). "
                f"Got: {header}"
            )

        rows = list(reader)

    n = len(rows)
    if n != 100:
        problems.append(f"Expected exactly 100 data rows, found {n}.")

    seen_ids = set()
    seen_ranks = set()
    dup_ids = set()
    dup_ranks = set()
    prev_score = None
    non_monotonic = False
    bad_rank_rows = []
    bad_score_rows = []

    for i, row in enumerate(rows, start=2):  # +1 header, +1 to be 1-indexed
        if len(row) < 3:
            problems.append(f"Row {i}: fewer than 3 columns.")
            continue
        cand_id, rank_s, score_s = row[0], row[1], row[2]

        if cand_id in seen_ids:
            dup_ids.add(cand_id)
        seen_ids.add(cand_id)

        try:
            rank = int(rank_s)
        except ValueError:
            bad_rank_rows.append(i)
            continue
        if rank in seen_ranks:
            dup_ranks.add(rank)
        seen_ranks.add(rank)

        try:
            score = float(score_s)
        except ValueError:
            bad_score_rows.append(i)
            continue
        if prev_score is not None and score > prev_score + 1e-9:
            non_monotonic = True
        prev_score = score

    if dup_ids:
        problems.append(f"Duplicate candidate_id values: {sorted(dup_ids)[:10]}"
                         f"{' ...' if len(dup_ids) > 10 else ''}")
    if dup_ranks:
        problems.append(f"Duplicate rank values: {sorted(dup_ranks)[:10]}"
                         f"{' ...' if len(dup_ranks) > 10 else ''}")
    if seen_ranks and (min(seen_ranks) != 1 or max(seen_ranks) != n):
        problems.append(
            f"Ranks must be exactly 1..{n} each used once. "
            f"Got min={min(seen_ranks) if seen_ranks else None}, "
            f"max={max(seen_ranks) if seen_ranks else None}."
        )
    if bad_rank_rows:
        problems.append(f"Non-integer rank at rows: {bad_rank_rows[:10]}")
    if bad_score_rows:
        problems.append(f"Non-numeric score at rows: {bad_score_rows[:10]}")
    if non_monotonic:
        problems.append(
            "score is not non-increasing as rank increases (rank 1 must "
            "have the highest score, rank 100 the lowest, ties allowed)."
        )

    return problems


def main():
    ap = argparse.ArgumentParser(description="Validate + finalize a FitHire submission CSV")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to the CSV produced by rank_cli.py")
    ap.add_argument("--participant-id", required=True, help="Your registered participant ID (no .csv extension)")
    ap.add_argument("--out-dir", default=".", help="Directory to write <participant_id>.csv into (default: cwd)")
    ap.add_argument("--force", action="store_true", help="Copy even if validation finds problems (not recommended)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"[FATAL] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[1] Validating {in_path} against submission_spec.md Section 3 rules...")
    problems = validate(in_path)

    if problems:
        print(f"\n[FAIL] {len(problems)} problem(s) found:")
        for p in problems:
            print(f"   - {p}")
        if not args.force:
            print("\nNot copying. Fix rank_cli.py's output and re-run, or pass "
                  "--force to copy anyway (not recommended — the portal "
                  "auto-validator will reject the same issues).")
            sys.exit(1)
        print("\n[WARN] --force set — copying despite the problems above.")
    else:
        print("    OK — 100 rows, ranks 1..100 each used once, unique "
              "candidate_ids, non-increasing score.")

    out_path = Path(args.out_dir) / f"{args.participant_id}.csv"
    shutil.copy2(in_path, out_path)
    print(f"\n[2] Copied to: {out_path.resolve()}")
    print("    This is the exact file to upload to the submission portal.")


if __name__ == "__main__":
    main()