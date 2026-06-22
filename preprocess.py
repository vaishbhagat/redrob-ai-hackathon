"""
preprocess.py — Offline pre-processing step.

Reads candidates.jsonl (streaming, line-by-line), extracts all 29 features
per candidate, and saves:
  - features.npy    : float32 array of shape (N, 29)
  - metadata.parquet: pandas DataFrame with human-readable columns for reasoning

Usage:
    python preprocess.py [--candidates PATH] [--out-dir DIR]

Defaults:
    --candidates  ./candidates.jsonl
    --out-dir     ./

This step is NOT timed; it runs once and can take a few minutes.
Memory usage stays well under 2 GB due to streaming.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# Import feature extractor
from features import extract_features


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Redrob candidate pre-processing")
    parser.add_argument(
        "--candidates",
        default="candidates.jsonl",
        help="Path to the candidates JSONL file (plain or .gz)",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory for features.npy and metadata.parquet",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    candidates_path = Path(args.candidates)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features_path  = out_dir / "features.npy"
    metadata_path  = out_dir / "metadata.parquet"

    print(f"[preprocess] Reading: {candidates_path}")
    print(f"[preprocess] Output:  {out_dir}")

    t0 = time.perf_counter()

    all_features: List[List[float]] = []
    all_meta: List[Dict] = []

    # Support both plain .jsonl and .jsonl.gz
    if candidates_path.suffix == ".gz":
        import gzip
        open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        open_fn = lambda p: open(p, "r", encoding="utf-8")

    n_processed = 0
    n_errors = 0

    with open_fn(candidates_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                n_errors += 1
                if n_errors <= 5:
                    print(f"  [WARN] JSON parse error on line {n_processed + 1}: {e}", file=sys.stderr)
                continue

            try:
                feats, meta = extract_features(candidate)
            except Exception as e:
                n_errors += 1
                if n_errors <= 5:
                    cid = candidate.get("candidate_id", "?")
                    print(f"  [WARN] Feature extraction failed for {cid}: {e}", file=sys.stderr)
                continue

            all_features.append(feats)
            all_meta.append(meta)
            n_processed += 1

            if n_processed % 10_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Processed {n_processed:,} candidates in {elapsed:.1f}s …")

    elapsed = time.perf_counter() - t0
    print(f"\n[preprocess] Done: {n_processed:,} candidates in {elapsed:.1f}s "
          f"({n_errors} errors)")

    # ── Save features array ──
    X = np.array(all_features, dtype=np.float32)
    print(f"[preprocess] Feature matrix shape: {X.shape}")
    np.save(features_path, X)
    print(f"[preprocess] Saved {features_path}  ({features_path.stat().st_size / 1e6:.1f} MB)")

    # ── Save metadata DataFrame ──
    meta_df = pd.DataFrame(all_meta)
    meta_df.to_parquet(metadata_path, index=False, engine="pyarrow")
    print(f"[preprocess] Saved {metadata_path}  ({metadata_path.stat().st_size / 1e6:.1f} MB)")

    print("\n[preprocess] Complete. Run rank.py next to generate submission.csv.")


if __name__ == "__main__":
    main()
