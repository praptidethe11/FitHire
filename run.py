#!/usr/bin/env python3
"""
FitHire — Quick Start
=========================
Run this file to start the FitHire server.
Then open http://localhost:8000 in your browser.
"""

import subprocess
import sys
import os


def install_deps():
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r",
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        "--quiet"
    ])


if __name__ == "__main__":
    try:
        import fastapi, uvicorn, rank_bm25
    except ImportError:
        install_deps()

    print("\n" + "=" * 55)
    print("  FitHire — Intelligent Candidate Ranking")
    print("=" * 55)
    print("  Open http://localhost:8000 in your browser")
    print("=" * 55 + "\n")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ])
    except KeyboardInterrupt:
        print("\n[INFO] FitHire server shut down gracefully.")