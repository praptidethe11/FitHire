#!/usr/bin/env python3
"""
FitHire — one-time model pre-download
======================================
Downloads and caches the two sentence-transformers models the ranking
pipeline depends on (bi-encoder + cross-encoder) BEFORE the timed ranking
step runs.

Why this exists:
  backend/main.py loads these models at import time:
      SentenceTransformer('all-MiniLM-L6-v2')
      CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  On first use, sentence-transformers fetches model weights from
  huggingface.co over the network. The competition's ranking step must run
  with no network access (submission_metadata_template.yaml declares
  has_network_during_ranking: false), so that download CANNOT happen during
  the timed run — it has to happen once, ahead of time, with this script.

Usage (run once, before you ever run rank_cli.py for a timed submission):
    python download_models.py

This populates the local HuggingFace cache (~/.cache/huggingface by default,
or wherever HF_HOME points). After this completes successfully, rank_cli.py
runs with HF_HUB_OFFLINE=1 and will use only these cached weights — it will
fail fast with a clear error if the cache is missing, instead of silently
hanging trying to reach the network.

If you're bundling model weights directly into your repo/sandbox image
(recommended for Stage 3 reproducibility — see submission_spec.md Section
10.3, "pre-computed artifacts... model weights, or a script that produces
them"), point HF_HOME at a folder inside the repo before running this, e.g.:
    HF_HOME=./model_cache python download_models.py
and then set the same HF_HOME when running rank_cli.py so it reads from the
bundled cache instead of the machine-wide one.
"""

import os
import sys
import time

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def main():
    print("=" * 60)
    print("  FitHire — pre-downloading semantic scoring models")
    print("=" * 60)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"\nHF_HOME is set — caching to: {hf_home}")
    else:
        print("\nHF_HOME not set — caching to the default HuggingFace "
              "cache directory (~/.cache/huggingface).")
        print("To bundle the cache inside this repo instead, re-run as:")
        print("    HF_HOME=./model_cache python download_models.py")

    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError:
        print(
            "\n[ERROR] sentence-transformers is not installed. "
            "Run `pip install -r requirements.txt` first."
        )
        sys.exit(1)

    t0 = time.time()
    print(f"\n[1/2] Downloading bi-encoder: {BI_ENCODER_MODEL} ...")
    SentenceTransformer(BI_ENCODER_MODEL, device="cpu")
    print(f"      Done in {time.time() - t0:.1f}s")

    t1 = time.time()
    print(f"\n[2/2] Downloading cross-encoder: {CROSS_ENCODER_MODEL} ...")
    CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    print(f"      Done in {time.time() - t1:.1f}s")

    print(f"\nAll models cached. Total time: {time.time() - t0:.1f}s")
    print(
        "\nYou can now run the ranking step offline. rank_cli.py sets "
        "HF_HUB_OFFLINE=1 automatically, so it will only ever read from "
        "this cache and will fail immediately (not hang) if a model is "
        "missing from it."
    )


if __name__ == "__main__":
    main()
