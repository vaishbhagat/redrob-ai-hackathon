"""
rank.py — Timed ranking script.

Loads pre-computed features.npy and metadata.parquet, computes final
scores, and outputs a compliant submission CSV.

Usage:
    python rank.py [--features features.npy] [--metadata metadata.parquet]
                   [--out submission.csv]

All operations are fully vectorised with NumPy — no Python loops over
the 100K candidate pool.  Typical runtime: < 5 seconds.
"""

from __future__ import annotations
import argparse
import time
import json
import gzip
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from config import (
    FEATURE_WEIGHTS,
    BEHAVIORAL_WEIGHTS,
    BEH_MOD_MIN,
    BEH_MOD_MAX,
    TITLE_CHASER_PENALTY,
    ALL_RESEARCH_PENALTY,
    CV_ONLY_PENALTY,
    ALL_CONSULTING_PENALTY,
)
from features import extract_features
from reasoning import generate_reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Feature index constants (keep in sync with features.py)
# ─────────────────────────────────────────────────────────────────────────────
IDX_SKILL_SCORE   = 0     # weighted skill match
IDX_SKILL_PROF    = 1     # avg proficiency of matched skills
IDX_SKILL_DUR     = 2     # avg normalised duration of matched skills
IDX_TITLE_ML      = 3     # ML-title count (normalised)
IDX_TITLE_SEN     = 4     # max seniority (normalised)
IDX_TITLE_CODING  = 5     # binary: currently in coding role
IDX_YOE_TOTAL     = 6     # total YOE (normalised)
IDX_YOE_ML        = 7     # ML-role years (normalised)
IDX_YOE_GAUSS     = 8     # Gaussian score around 7 yrs
IDX_COMP_PRODUCT  = 9     # binary: current company is product co.
IDX_COMP_CONS_ALL = 10    # binary: all career at consulting firms
IDX_COMP_FAANG    = 11    # binary: ever at FAANG/unicorn
IDX_COMP_DISTINCT = 12    # normalised distinct company count
IDX_EDU_TIER      = 13    # institution tier score
IDX_EDU_CS        = 14    # binary: CS/ML degree
IDX_LOC_INDIA     = 15    # binary: in India
IDX_LOC_RELOCATE  = 16    # binary: willing to relocate
IDX_LOC_BONUS     = 17    # location bonus (0.2 / 0.8 / 1.0)
IDX_RF_CHASER     = 18    # red flag: title-chaser
IDX_RF_RESEARCH   = 19    # red flag: all-research career
IDX_RF_CV_ONLY    = 20    # red flag: CV/Speech without NLP/IR
IDX_BEH_START     = 21    # first behavioral feature index
IDX_BEH_END       = 28    # last behavioral feature index (exclusive)
IDX_HONEYPOT      = 28    # honeypot flag


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(X: np.ndarray) -> np.ndarray:
    """
    Compute final ranking scores for all candidates.

    Algorithm:
      1. Base fit score  = dot(features[0:18], weights[0:18])
      2. Behavioral modifier = weighted sum of features[21:28],
                               scaled to [BEH_MOD_MIN, BEH_MOD_MAX]
      3. Penalty multipliers from red-flag features [18, 19, 20]
      4. All-consulting penalty from feature 10
      5. Honeypot flag (28) → score set to -1e9

    All operations are vectorised (no Python loops over rows).
    """
    N = X.shape[0]

    # ── 1. Base fit score (features 0–17 only) ──
    weights = np.array(FEATURE_WEIGHTS, dtype=np.float32)
    base_weights = weights.copy()
    base_weights[IDX_BEH_START:]  = 0.0  # zero out behavioral + beyond
    base_weights[IDX_RF_CHASER]   = 0.0  # red flags don't add
    base_weights[IDX_RF_RESEARCH] = 0.0
    base_weights[IDX_RF_CV_ONLY]  = 0.0

    base_score = X[:, :len(base_weights)] @ base_weights  # shape (N,)

    # ── 2. Behavioral modifier ──
    beh_weights = np.array(BEHAVIORAL_WEIGHTS, dtype=np.float32)
    beh_features = X[:, IDX_BEH_START:IDX_BEH_END]       # (N, 7)
    beh_raw = beh_features @ beh_weights                   # (N,)
    # Scale raw [0,1] → [BEH_MOD_MIN, BEH_MOD_MAX]
    beh_mod = BEH_MOD_MIN + beh_raw * (BEH_MOD_MAX - BEH_MOD_MIN)

    # ── 3. Red-flag penalty multipliers ──
    pen = np.ones(N, dtype=np.float32)
    pen[X[:, IDX_RF_CHASER]   > 0.5] *= TITLE_CHASER_PENALTY
    pen[X[:, IDX_RF_RESEARCH] > 0.5] *= ALL_RESEARCH_PENALTY
    pen[X[:, IDX_RF_CV_ONLY]  > 0.5] *= CV_ONLY_PENALTY
    pen[X[:, IDX_COMP_CONS_ALL] > 0.5] *= ALL_CONSULTING_PENALTY

    # ── 4. Final score ──
    final_score = base_score * beh_mod * pen

    # ── 5. Honeypot override ──
    final_score[X[:, IDX_HONEYPOT] > 0.5] = -1e9

    return final_score


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--features",  default="features.npy",        help="Pre-computed feature matrix")
    parser.add_argument("--metadata",  default="metadata.parquet",     help="Pre-computed metadata")
    parser.add_argument("--out",       default="submission.csv",       help="Output CSV path")
    parser.add_argument("--top-n",     type=int, default=100,          help="Number of candidates to rank")
    # Optional: candidates file (ignored at rank time; kept for CLI compatibility)
    parser.add_argument("--candidates", default=None, help="(ignored at rank time)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.perf_counter()

    out_path       = Path(args.out)
    top_n          = args.top_n

    # Check if candidates file is provided and exists
    if args.candidates and Path(args.candidates).exists():
        candidates_path = Path(args.candidates)
        print(f"[rank] Processing candidates on-the-fly from: {candidates_path}")
        
        # Support both plain .jsonl and .jsonl.gz
        if candidates_path.suffix == ".gz":
            open_fn = lambda p: gzip.open(p, "rt", encoding="utf-8")
        else:
            open_fn = lambda p: open(p, "r", encoding="utf-8")

        all_features = []
        all_meta = []
        n_processed = 0
        n_errors = 0

        with open_fn(candidates_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                    feats, meta = extract_features(candidate)
                    all_features.append(feats)
                    all_meta.append(meta)
                except Exception:
                    n_errors += 1
                    continue
                n_processed += 1

        print(f"[rank] Processed {n_processed} candidates on-the-fly ({n_errors} errors)")
        X = np.array(all_features, dtype=np.float32)
        meta_df = pd.DataFrame(all_meta)
    else:
        # Fall back to pre-computed files
        features_path  = Path(args.features)
        metadata_path  = Path(args.metadata)
        
        print(f"[rank] Loading {features_path} ...")
        X = np.load(features_path)
        print(f"[rank] Feature matrix: {X.shape}")

        print(f"[rank] Loading {metadata_path} ...")
        meta_df = pd.read_parquet(metadata_path, engine="pyarrow")
        print(f"[rank] Metadata rows: {len(meta_df)}")

        if len(meta_df) != X.shape[0]:
            raise ValueError(
                f"Mismatch: {X.shape[0]} feature rows vs {len(meta_df)} metadata rows"
            )

    # ── Compute scores ──
    print("[rank] Computing scores ...")
    scores = compute_scores(X)

    # ── Select top-N with proper tie-breaking ──
    # Sort by score descending and candidate_id ascending
    df_temp = pd.DataFrame({
        'idx': np.arange(len(scores)),
        'score': scores,
        'candidate_id': meta_df['candidate_id']
    })
    df_temp = df_temp.sort_values(by=['score', 'candidate_id'], ascending=[False, True])
    top_idx = df_temp['idx'].values[:top_n]

    # Build result rows
    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        meta = meta_df.iloc[idx].to_dict()
        score = float(scores[idx])
        reasoning = generate_reasoning(meta, score, rank)
        rows.append({
            "candidate_id": meta["candidate_id"],
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

    # ── Write CSV ──
    result_df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    result_df.to_csv(out_path, index=False, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"\n[rank] Done in {elapsed:.2f}s -> {out_path}")
    print(f"[rank] Score range: {result_df['score'].min():.4f} - {result_df['score'].max():.4f}")
    print(f"[rank] Top-5 candidates:")
    for _, row in result_df.head(5).iterrows():
        print(f"  #{row['rank']:3d}  {row['candidate_id']}  score={row['score']:.4f}")


if __name__ == "__main__":
    main()
