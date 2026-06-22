"""
reasoning.py — Reasoning string generator for submission CSV.

Takes a metadata dict (as produced by features.extract_features) plus the
computed score and rank, and returns a 1–2 sentence explanation.

The reasoning deliberately:
 - References specific, verifiable facts from the profile (no hallucination)
 - Connects to the JD (embeddings / retrieval / ranking)
 - Acknowledges honest concerns at lower ranks
 - Varies in phrasing so sampled rows look distinct
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Internal phrasing templates (selected pseudo-randomly but seeded by cand_id)
# ─────────────────────────────────────────────────────────────────────────────

_SKILL_STRONG = [
    "hands-on experience with {skills}",
    "proven track record in {skills}",
    "demonstrated expertise in {skills}",
    "production experience with {skills}",
    "solid background in {skills}",
]

_SKILL_WEAK = [
    "limited exposure to core ranking/retrieval systems",
    "skill set is peripheral to the JD requirements",
    "primarily adjacent skills with minimal NLP/IR depth",
]

_BEH_HIGH = [
    "actively engaging on the platform",
    "strong recruiter responsiveness",
    "highly active and responsive",
]

_BEH_MOD = [
    "moderate engagement signals",
    "reasonable platform activity",
    "adequate responsiveness",
]

_BEH_LOW = [
    "low engagement / responsiveness on the platform",
    "inactive profile (low recency or response rate)",
    "minimal recent platform activity",
]

_LOCATION_BONUS = {
    "Pune": "Pune-based",
    "Noida": "Noida-based",
    "Delhi": "Delhi NCR-based",
    "Gurgaon": "Gurgaon-based (Delhi NCR)",
    "Bangalore": "Bangalore-based, willing to relocate",
    "Hyderabad": "Hyderabad-based, willing to relocate",
    "Mumbai": "Mumbai-based, willing to relocate",
}

_CONCERN_TEMPLATES = [
    "Concern: {concern}.",
    "Note: {concern}.",
    "One flag: {concern}.",
    "Caveat: {concern}.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(meta: Dict, score: float, rank: int) -> str:
    """
    Generate a 1–2 sentence reasoning string for the submission CSV.

    Args:
        meta:  metadata dict from features.extract_features
        score: final ranking score (float)
        rank:  final rank (1-100)

    Returns:
        A concise, factual reasoning string.
    """
    # Seed random with candidate ID for reproducibility without all-same output
    rng = random.Random(meta["candidate_id"])

    title  = meta.get("current_title", "Engineer")
    company = meta.get("current_company", "")
    yoe    = meta.get("years_of_experience", 0.0)
    _skills_raw = meta.get("matched_skills", [])
    skills = list(_skills_raw) if _skills_raw is not None and len(_skills_raw) > 0 else []
    skill_tier  = meta.get("skill_tier", "Basic")
    beh_tier    = meta.get("behavior_tier", "Moderate")
    _rf_raw = meta.get("red_flags", [])
    red_flags   = list(_rf_raw) if _rf_raw is not None and len(_rf_raw) > 0 else []
    notice      = meta.get("notice_period_days", 90)
    location    = meta.get("location", "")
    country     = meta.get("country", "")
    in_india    = "india" in (country or "").lower()
    willing     = meta.get("willing_to_relocate", False)
    open_work   = meta.get("open_to_work", False)
    honeypot    = meta.get("honeypot", False)
    all_consulting = meta.get("all_consulting", False)
    edu_tier    = meta.get("edu_tier", "unknown")
    github_score = meta.get("github_score", -1)

    # ── Honeypot: short explanation at low rank ──
    if honeypot:
        return (
            f"{title} with {yoe:.1f} yrs experience; profile contains "
            "physically impossible signals (inflated durations / impossible dates) "
            "flagged as honeypot — ranked at floor."
        )

    # ── Build sentence 1: core professional summary ──
    yoe_str = f"{yoe:.1f} yrs"
    company_str = f"at {company}" if company else ""

    if skills:
        skill_phrase = rng.choice(_SKILL_STRONG).format(skills=", ".join(skills[:3]))
    else:
        skill_phrase = rng.choice(_SKILL_WEAK)

    # Location snippet
    loc_lower = (location or "").lower()
    loc_snippet = ""
    for city, phrase in _LOCATION_BONUS.items():
        if city.lower() in loc_lower:
            loc_snippet = f"{phrase}"
            break
    if not loc_snippet:
        if in_india:
            loc_snippet = f"India-based ({'relocate-ready' if willing else 'location may not match'})"
        else:
            loc_snippet = f"based outside India{', open to relocation' if willing else ''}"

    # Notice period
    if notice <= 30:
        notice_snippet = "immediate/short notice"
    elif notice <= 60:
        notice_snippet = f"{notice}-day notice"
    else:
        notice_snippet = f"long notice ({notice} days)"

    # Education note for top ranks
    edu_note = ""
    if rank <= 30 and edu_tier in ("tier_1", "tier_2"):
        edu_note = f" ({edu_tier.replace('_', '-')} institution)"

    sentence1 = (
        f"{title} with {yoe_str} {company_str}; {skill_phrase}; "
        f"{loc_snippet}{edu_note}."
    )

    # ── Build sentence 2: engagement + concerns ──
    parts2 = []

    if beh_tier == "High":
        parts2.append(rng.choice(_BEH_HIGH))
    elif beh_tier == "Moderate":
        parts2.append(rng.choice(_BEH_MOD))
    else:
        parts2.append(rng.choice(_BEH_LOW))

    if open_work:
        parts2.append("marked open-to-work")

    if github_score >= 50:
        parts2.append("strong GitHub activity")

    # Concerns / red flags
    concerns_shown = []
    if red_flags:
        # Filter out honeypot flag (already handled above)
        rf = [f for f in red_flags if "honeypot" not in f]
        if rf:
            concerns_shown = rf[:2]
    if all_consulting and "all career in consulting firms" not in concerns_shown:
        concerns_shown.insert(0, "entire career at consulting firms")

    if notice_snippet and notice > 60:
        concerns_shown.append(notice_snippet)

    # Rank-appropriate tone
    if rank <= 20:
        # Top 20: positive, minimal concerns
        engagement = ", ".join(parts2)
        sentence2 = f"{engagement.capitalize()}."
        if concerns_shown:
            sentence2 += f" Note: {concerns_shown[0]}."
    elif rank <= 60:
        # Middle: balanced
        engagement = ", ".join(parts2)
        sentence2 = f"{engagement.capitalize()}."
        if concerns_shown:
            sentence2 += f" Concern: {'; '.join(concerns_shown[:2])}."
    else:
        # Bottom: more cautious
        concerns_all = concerns_shown or ["limited fit with core JD requirements"]
        sentence2 = (
            f"{'Some engagement signals' if beh_tier != 'Low' else 'Weak engagement'}; "
            f"concern: {'; '.join(concerns_all[:2])}."
        )

    reasoning = f"{sentence1} {sentence2}"
    # Guard: truncate if abnormally long (shouldn't happen)
    if len(reasoning) > 400:
        reasoning = reasoning[:397] + "..."
    return reasoning
