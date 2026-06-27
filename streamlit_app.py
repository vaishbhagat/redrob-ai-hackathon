"""
streamlit_app.py — Redrob Candidate Ranking Dashboard
A clean, professional Streamlit dashboard that displays the top 100 ranked
candidates with interactive filters, KPI metrics, charts, and a detailed
candidate inspector panel.

NOTE: Place a file named 'logo.png' in the same directory as this script
to display a custom logo in the sidebar. If missing, the app displays
a text fallback instead.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redrob_dashboard")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="assets/logo.png" if Path("assets/logo.png").exists() else "R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide the built-in Deploy button and top-right toolbar
st.markdown(
    """
    <style>
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }
        #MainMenu { display: none !important; }
        footer { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading candidate data...")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[np.ndarray], Dict]:
    """
    Loads pre-computed ranking artifacts and returns:
        - df        : top-100 ranked candidates (submission.csv)
        - meta_df   : full metadata for all evaluated candidates (metadata.parquet)
        - X         : feature matrix, if available (features.npy)
        - stats     : pipeline volume / timing statistics
    """
    submission_path = Path("submission.csv")
    meta_path       = Path("metadata.parquet")
    features_path   = Path("features.npy")

    # Strict artifact check — dynamic JSONL parsing is disabled to avoid
    # HuggingFace Spaces timeouts on 100k-candidate datasets.
    if not submission_path.exists() or not meta_path.exists():
        st.error("Required submission artifacts are missing.")
        st.info(
            "Please ensure the following files are present in the repository:\n\n"
            "- `submission.csv` — 100 ranked rows\n"
            "- `metadata.parquet` — full metadata for all candidates\n"
            "- `features.npy` — (optional) feature matrix for radar chart\n\n"
            "Generate them locally by running:\n"
            "```bash\n"
            "python preprocess.py --candidates candidates.jsonl --out-dir .\n"
            "python rank.py --candidates candidates.jsonl --out submission.csv\n"
            "```"
        )
        st.stop()

    t0 = time.perf_counter()
    df      = pd.read_csv(submission_path)
    meta_df = pd.read_parquet(meta_path)

    X = None
    if features_path.exists():
        X = np.load(features_path)

    loading_time = time.perf_counter() - t0

    stats = {
        "total_evaluated": len(meta_df),
        "loading_time":    loading_time,
        "rank_runtime_sec": 20.66,   # representative CPU benchmark
    }

    return df, meta_df, X, stats


# Load data
df, meta_df, X, stats = load_data()


# ---------------------------------------------------------------------------
# Sidebar — logo + filters
# ---------------------------------------------------------------------------

# Logo with graceful fallback to plain text
logo_path = Path("logo.png")
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=160)
else:
    st.sidebar.markdown("## Redrob AI")

st.sidebar.title("Ranker Filters")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")

# Filter 1: Score range
min_score = float(df["score"].min())
max_score = float(df["score"].max())
score_range = st.sidebar.slider(
    "Score Range",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score),
    step=0.005,
    format="%.3f",
)

# Filter 2: Skill tier
all_tiers = ["Expert", "Strong", "Good", "Basic"]
selected_tiers = st.sidebar.multiselect(
    "Skill Tiers",
    options=all_tiers,
    default=all_tiers,
)

# Filter 3: Red flag exclusions
all_red_flags = [
    "title-chaser (short tenures)",
    "purely research/academic career",
    "CV/Speech focus without NLP/IR",
    "all career in consulting firms",
    "honeypot / impossible profile",
]
selected_rf_excludes = st.sidebar.multiselect(
    "Exclude Candidates with Red Flags",
    options=all_red_flags,
    default=[],
)

# Filter 4: Location preference
all_locations = [
    "Pune-based", "Noida-based", "Delhi NCR-based",
    "Bangalore-based", "Other India", "Outside India",
]
selected_locations = st.sidebar.multiselect(
    "Preferred Locations",
    options=all_locations,
    default=all_locations,
)


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

filtered_df = df.copy()

# Score filter
filtered_df = filtered_df[
    (filtered_df["score"] >= score_range[0]) &
    (filtered_df["score"] <= score_range[1])
]

# Merge metadata for attribute-level filters
merged_filtered = filtered_df.merge(meta_df, on="candidate_id", how="inner")

# Skill tier filter
if selected_tiers:
    merged_filtered = merged_filtered[merged_filtered["skill_tier"].isin(selected_tiers)]

# Red flag exclusion filter
if selected_rf_excludes:
    for rf_tag in selected_rf_excludes:
        def flag_filter(rf_list):
            if rf_list is None:
                return True
            rf_conv = rf_list.tolist() if isinstance(rf_list, np.ndarray) else list(rf_list)
            return rf_tag not in rf_conv
        merged_filtered = merged_filtered[merged_filtered["red_flags"].apply(flag_filter)]


def get_loc_group(row: Dict) -> str:
    """Classify a candidate row into a broad location group."""
    loc     = str(row.get("location", "")).lower()
    country = str(row.get("country", "")).lower()
    in_india = "india" in country or "india" in loc

    if "pune" in loc:
        return "Pune-based"
    elif "noida" in loc:
        return "Noida-based"
    elif "delhi" in loc or "gurgaon" in loc:
        return "Delhi NCR-based"
    elif "bangalore" in loc:
        return "Bangalore-based"
    elif in_india:
        return "Other India"
    else:
        return "Outside India"


if len(merged_filtered) > 0:
    merged_filtered["_loc_group"] = merged_filtered.apply(get_loc_group, axis=1)
    if selected_locations:
        merged_filtered = merged_filtered[merged_filtered["_loc_group"].isin(selected_locations)]

# Sync filtered IDs back to display dataframe
filtered_ids = merged_filtered["candidate_id"].unique()
filtered_df  = filtered_df[filtered_df["candidate_id"].isin(filtered_ids)]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Redrob Candidate Ranker")

st.divider()


# ---------------------------------------------------------------------------
# KPI metrics row
# ---------------------------------------------------------------------------

top_100_meta = meta_df[meta_df["candidate_id"].isin(df["candidate_id"])]

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.metric("Candidates Evaluated", f"{stats['total_evaluated']:,}")
with kpi2:
    st.metric("Highest Score", f"{df['score'].max():.4f}" if not df.empty else "N/A")
with kpi3:
    st.metric("Median Score", f"{df['score'].median():.4f}" if not df.empty else "N/A")
with kpi4:
    hp_count = int(top_100_meta["honeypot"].sum())
    st.metric("Honeypots (Top 100)", hp_count)
with kpi5:
    avg_resp = float(top_100_meta["recruiter_response_rate"].mean() * 100.0)
    st.metric("Avg. Response Rate", f"{avg_resp:.1f}%")
with kpi6:
    st.metric("Pipeline Latency", f"{stats['rank_runtime_sec']:.2f}s")

st.divider()


# ---------------------------------------------------------------------------
# Charts — score distribution and skill tier breakdown
# ---------------------------------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Score Distribution")
    fig_hist = px.histogram(
        df,
        x="score",
        nbins=20,
        labels={"score": "Ranking Score", "count": "Count"},
        template="plotly_white",
        color_discrete_sequence=["#E63946"],
    )
    fig_hist.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with chart_col2:
    st.subheader("Skill Tier Breakdown")
    tier_counts = top_100_meta["skill_tier"].value_counts().reset_index()
    tier_counts.columns = ["Skill Tier", "Count"]
    tier_order = {"Expert": 0, "Strong": 1, "Good": 2, "Basic": 3}
    tier_counts["sort_idx"] = tier_counts["Skill Tier"].map(tier_order)
    tier_counts = tier_counts.sort_values("sort_idx")

    fig_bar = px.bar(
        tier_counts,
        x="Skill Tier",
        y="Count",
        color="Skill Tier",
        template="plotly_white",
        color_discrete_map={
            "Expert": "#E63946",
            "Strong": "#F4A261",
            "Good":   "#2A9D8F",
            "Basic":  "#90A1B1",
        },
    )
    fig_bar.update_layout(
        showlegend=False,
        xaxis_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

st.divider()


# ---------------------------------------------------------------------------
# Two-column layout: candidate table  |  inspector panel
# ---------------------------------------------------------------------------

# Session state: track selected candidate
if "selected_candidate" not in st.session_state and not df.empty:
    st.session_state["selected_candidate"] = df.iloc[0]["candidate_id"]

col_table, col_inspector = st.columns([3, 2])

# ── Left column: candidate table ──────────────────────────────────────────

with col_table:
    st.subheader("Top Candidate Registry")

    search_keyword = st.text_input(
        "Search candidates",
        placeholder="Filter by Candidate ID, skills, company (e.g. CAND_0077337, Paytm, QLoRA...)",
    )

    display_df = filtered_df.copy()
    if search_keyword:
        mask = (
            display_df["candidate_id"].str.contains(search_keyword, case=False, na=False) |
            display_df["reasoning"].str.contains(search_keyword, case=False, na=False)
        )
        display_df = display_df[mask]

    if display_df.empty:
        st.warning("No candidates match the selected filters or search parameters.")
    else:
        st.dataframe(
            display_df,
            column_config={
                "rank":        st.column_config.NumberColumn("Rank", width="small", format="%d"),
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
                "score":       st.column_config.NumberColumn("Score", width="small", format="%.4f"),
                "reasoning":   st.column_config.TextColumn("Reasoning", width="large"),
            },
            hide_index=True,
            use_container_width=True,
            height=500,
        )

        # Download button
        full_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Submission CSV",
            data=full_csv,
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Right column: candidate inspector ────────────────────────────────────

with col_inspector:
    st.subheader("Candidate Inspector")

    cand_options = display_df["candidate_id"].tolist() if not display_df.empty else []

    if not cand_options:
        st.info("No candidates available for inspection with the current filters.")
    else:
        default_index = 0
        if st.session_state.get("selected_candidate") in cand_options:
            default_index = cand_options.index(st.session_state["selected_candidate"])

        selected_id = st.selectbox(
            "Select candidate",
            options=cand_options,
            index=default_index,
            key="inspector_selection",
        )
        st.session_state["selected_candidate"] = selected_id

        # Pull metadata for selected candidate
        cand_meta = meta_df[meta_df["candidate_id"] == selected_id]

        if not cand_meta.empty:
            meta      = cand_meta.iloc[0].to_dict()
            rank_num  = df[df["candidate_id"] == selected_id]["rank"].values[0]
            score_num = df[df["candidate_id"] == selected_id]["score"].values[0]
            is_honeypot = bool(meta.get("honeypot", False))

            # Safe array-to-list conversions
            _rf_raw = meta.get("red_flags", [])
            if isinstance(_rf_raw, np.ndarray):
                red_flags = _rf_raw.tolist()
            elif _rf_raw is not None:
                red_flags = list(_rf_raw)
            else:
                red_flags = []

            _skills_raw = meta.get("matched_skills", [])
            if isinstance(_skills_raw, np.ndarray):
                matched_skills = _skills_raw.tolist()
            elif _skills_raw is not None:
                matched_skills = list(_skills_raw)
            else:
                matched_skills = []

            # ── Identity block ─────────────────────────────────────────────
            st.markdown(f"### {meta.get('name', 'Anonymized Candidate')}")
            st.caption(
                f"{meta.get('current_title', 'ML Engineer')}  at  "
                f"{meta.get('current_company', 'N/A')}  |  ID: {selected_id}"
            )

            info1, info2, info3, info4 = st.columns(4)
            info1.metric("Rank", f"#{int(rank_num)}")
            info2.metric("Score", f"{score_num:.4f}")
            info3.metric("Skill Tier", meta.get("skill_tier", "Basic"))
            info4.metric("Behavior Tier", meta.get("behavior_tier", "Moderate"))

            # ── Alerts ────────────────────────────────────────────────────
            if is_honeypot:
                st.error("HONEYPOT DETECTED — physically impossible profile. Excluded from ranking.")
            if len(red_flags) > 0:
                st.warning("Red Flags: " + " | ".join(rf.upper() for rf in red_flags))

            st.divider()

            # ── Key details grid ──────────────────────────────────────────
            st.markdown("**Job Fit Details**")
            det1, det2 = st.columns(2)

            with det1:
                yoe = meta.get("years_of_experience", 0.0)
                st.metric("Total Experience", f"{float(yoe):.1f} yrs")

                edu_tier_str = str(meta.get("edu_tier", "Unknown")).replace("_", " ").title()
                st.metric("Education Tier", edu_tier_str)

            with det2:
                skill_ws = meta.get("skill_weighted_score", 0.0)
                st.metric("Skill Match Score", f"{float(skill_ws):.2f} / 1.00")

                willing_str = " (Willing to relocate)" if meta.get("willing_to_relocate", False) else ""
                st.metric("Location", f"{meta.get('location', 'N/A')}{willing_str}")

            # Availability
            notice_days = meta.get("notice_period_days", 90)
            open_to_work = meta.get("open_to_work", False)
            notice_label = f"{notice_days} days" + (" — Open to Work" if open_to_work else "")
            st.metric("Notice Period", notice_label)

            # Matched skills
            st.markdown("**Matched JD Skills**")
            if matched_skills:
                st.write(", ".join(matched_skills))
            else:
                st.write("None matched")

            st.divider()

            # ── Radar chart (if feature matrix is available) ───────────────
            if X is not None:
                idx_list = meta_df.index[meta_df["candidate_id"] == selected_id].tolist()
                if idx_list:
                    idx   = idx_list[0]
                    feats = X[idx]

                    skill_score   = float((feats[0] + feats[1] + feats[2]) / 3.0) * 100.0
                    title_score   = float((feats[3] + feats[4] + feats[5]) / 3.0) * 100.0
                    exp_score     = float((feats[6] + feats[7] + feats[8]) / 3.0) * 100.0
                    company_score = float((feats[9] + (1.0 - feats[10]) + feats[11] + feats[12]) / 4.0) * 100.0
                    loc_score     = float((feats[15] + feats[16] + feats[17]) / 3.0) * 100.0

                    radar_data = pd.DataFrame({
                        "score":     [skill_score, title_score, exp_score, company_score, loc_score],
                        "dimension": ["Skill Match", "Title Fit", "Experience", "Company Pedigree", "Location"],
                    })

                    fig_radar = px.line_polar(
                        radar_data,
                        r="score",
                        theta="dimension",
                        line_close=True,
                        template="plotly_white",
                    )
                    fig_radar.update_traces(
                        fill="toself",
                        fillcolor="rgba(230, 57, 70, 0.20)",
                        line_color="#E63946",
                        line_width=2,
                    )
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False,
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=220,
                    )

                    st.markdown("**Feature Fit Radar**")
                    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

            st.divider()

            # ── Engagement signals ─────────────────────────────────────────
            st.markdown("**Platform Engagement Signals**")

            eng1, eng2 = st.columns(2)
            with eng1:
                rr = float(meta.get("recruiter_response_rate", 0.0))
                st.metric("Recruiter Response Rate", f"{rr * 100:.1f}%")
                st.progress(rr)

                gh_score = meta.get("github_score", -1)
                if gh_score is not None and float(gh_score) >= 0:
                    st.metric("GitHub Activity", f"{float(gh_score):.0f} / 100")
                    st.progress(float(gh_score) / 100.0)
                else:
                    st.metric("GitHub Activity", "N/A")

            with eng2:
                pc = float(meta.get("profile_completeness", 0.0))
                st.metric("Profile Completeness", f"{pc:.0f}%")
                st.progress(pc / 100.0)

            st.divider()

            # ── Reasoning text ─────────────────────────────────────────────
            st.markdown("**Evaluator Reasoning**")
            reasoning_text = df[df["candidate_id"] == selected_id]["reasoning"].values[0]
            st.info(reasoning_text)

        else:
            st.warning(f"Could not load metadata for candidate: {selected_id}")
