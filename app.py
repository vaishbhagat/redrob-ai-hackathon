"""
app.py — Sandbox / demo entry point for HuggingFace Spaces (or local demo).

Demonstrates the full ranking pipeline on a small sample (≤100 candidates).
Can be run locally as:  python app.py
Or adapted for Streamlit / Gradio if hosted on HuggingFace Spaces.

In production mode, this file is NOT used — only rank.py is called.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from features import extract_features
from rank import compute_scores
from reasoning import generate_reasoning


SAMPLE_CANDIDATES_PATH = Path("sample_candidates.json")
CANDIDATES_JSONL_PATH  = Path("candidates.jsonl")


def load_sample_candidates(path: Path, max_n: int = 100) -> List[Dict]:
    """Load up to max_n candidates from a JSON array or JSONL file."""
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("["):
        candidates = json.loads(content)
        return candidates[:max_n]
    else:
        candidates = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass
            if len(candidates) >= max_n:
                break
        return candidates


def run_pipeline(candidates: List[Dict], top_n: int = 100) -> pd.DataFrame:
    """Run full feature extraction + scoring on a list of candidates."""
    top_n = min(top_n, len(candidates))

    all_features = []
    all_meta = []
    for c in candidates:
        feats, meta = extract_features(c)
        all_features.append(feats)
        all_meta.append(meta)

    X = np.array(all_features, dtype=np.float32)
    scores = compute_scores(X)

    sorted_idx = np.argsort(-scores, kind="stable")
    top_idx = sorted_idx[:top_n]

    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        meta = all_meta[idx]
        score = float(scores[idx])
        reasoning = generate_reasoning(meta, score, rank)
        rows.append({
            "candidate_id": meta["candidate_id"],
            "rank":         rank,
            "score":        round(score, 6),
            "reasoning":    reasoning,
        })

    return pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])


def main():
    print("=" * 60)
    print("Redrob Candidate Ranking - Sandbox Demo")
    print("=" * 60)

    # Try to load sample candidates
    if SAMPLE_CANDIDATES_PATH.exists():
        print(f"\nLoading candidates from {SAMPLE_CANDIDATES_PATH} ...")
        candidates = load_sample_candidates(SAMPLE_CANDIDATES_PATH, max_n=100)
    elif CANDIDATES_JSONL_PATH.exists():
        print(f"\nLoading first 100 candidates from {CANDIDATES_JSONL_PATH} ...")
        candidates = []
        with open(CANDIDATES_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except Exception:
                        pass
                if len(candidates) >= 100:
                    break
    else:
        print("ERROR: No candidate data found. Please supply sample_candidates.json or candidates.jsonl")
        sys.exit(1)

    print(f"Loaded {len(candidates)} candidates.")

    t0 = time.perf_counter()
    top_n = min(100, len(candidates))
    result = run_pipeline(candidates, top_n=top_n)
    elapsed = time.perf_counter() - t0

    print(f"\nRanking complete in {elapsed:.3f}s")
    print(f"\nTop 10 candidates:\n")
    print(result.head(10).to_string(index=False))

    out_path = Path("demo_submission.csv")
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved demo submission -> {out_path}")

    # Validate quickly
    from validate_submission import validate_submission
    errs = validate_submission(str(out_path))
    if errs:
        print("\nValidation issues:")
        for e in errs:
            print(f"  - {e}")
    else:
        print("Validation: PASSED [OK]")


if __name__ == "__main__":
    main()
