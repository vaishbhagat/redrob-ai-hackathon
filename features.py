"""
features.py — Feature extraction helpers for each candidate record.

Each public function takes a parsed candidate dict and returns one or more
float feature values.  All outputs are in [0, 1] unless noted.
"""

from __future__ import annotations
import re
import math
import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from config import (
    JD_SKILLS,
    ML_TITLE_KEYWORDS,
    SENIORITY_LEVELS,
    CODING_TITLES,
    NON_CODING_TITLES,
    CONSULTING_FIRMS,
    PRODUCT_COMPANIES,
    EDU_TIER_SCORES,
    CS_ML_FIELDS,
    PREFERRED_CITIES,
    INDIA_COUNTRY_NAMES,
    RESEARCH_TITLES,
    CV_SPEECH_ROBOTICS_SKILLS,
    NLP_IR_SKILLS,
    PROFICIENCY_WEIGHTS,
    SKILL_DURATION_CAP,
    YOE_GAUSSIAN_MU,
    YOE_GAUSSIAN_SIGMA,
    REFERENCE_DATE,
    NOTICE_30,
    NOTICE_60,
    NOTICE_90,
    NOTICE_120,
    RECENCY_30,
    RECENCY_90,
    JOB_DURATION_CALENDAR_SLACK,
    EXPERT_ZERO_DUR_THRESHOLD,
    ENDORSEMENTS_PER_MONTH_LIMIT,
    COMPANY_FOUNDING_YEARS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(value: float, lo: float, hi: float) -> float:
    """Clamp and min-max normalise to [0, 1]."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _lower(s: str) -> str:
    return s.lower().strip() if s else ""


def _skill_matches_jd(skill_name: str) -> bool:
    """Return True if the skill name matches any JD keyword."""
    sn = _lower(skill_name)
    for kw in JD_SKILLS:
        if kw in sn or sn in kw:
            return True
    return False


def _title_has_ml(title: str) -> bool:
    t = _lower(title)
    for kw in ML_TITLE_KEYWORDS:
        if kw in t:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 1: Skill Match (features 0, 1, 2)
# ─────────────────────────────────────────────────────────────────────────────

def skill_features(skills: List[Dict]) -> Tuple[float, float, float]:
    """
    Returns:
        f0: weighted_overlap_score   – main skill relevance signal
        f1: avg_proficiency_matched  – average proficiency weight of matched skills
        f2: avg_duration_matched     – average normalised duration of matched skills
    """
    total_skills = max(len(skills), 1)
    matched_weighted_sum = 0.0
    matched_prof_sum = 0.0
    matched_dur_sum = 0.0
    matched_count = 0

    for s in skills:
        if _skill_matches_jd(s.get("name", "")):
            prof_w = PROFICIENCY_WEIGHTS.get(s.get("proficiency", "beginner"), 0.3)
            dur = min(s.get("duration_months", 0) or 0, SKILL_DURATION_CAP)
            dur_norm = dur / SKILL_DURATION_CAP
            matched_weighted_sum += prof_w * (0.5 + 0.5 * dur_norm)
            matched_prof_sum += prof_w
            matched_dur_sum += dur_norm
            matched_count += 1

    # Normalise weighted sum by total skills to penalise long skill lists padded with noise
    f0 = min(matched_weighted_sum / (total_skills ** 0.5), 1.0)  # sqrt dampening
    f1 = matched_prof_sum / matched_count if matched_count > 0 else 0.0
    f2 = matched_dur_sum / matched_count if matched_count > 0 else 0.0
    return f0, f1, f2


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 2: Title Fit (features 3, 4, 5)
# ─────────────────────────────────────────────────────────────────────────────

def _seniority_of_title(title: str) -> int:
    t = _lower(title)
    for level in [3, 2, 1, 0]:
        for kw in SENIORITY_LEVELS[level]:
            if kw in t:
                return level
    return 1  # default: mid-level


def title_features(profile: Dict, career_history: List[Dict]) -> Tuple[float, float, float]:
    """
    f3: normalised count of ML-relevant job titles across full career
    f4: normalised max seniority (0-3 → 0.0-1.0)
    f5: binary – current title is a coding/technical role
    """
    all_titles = [profile.get("current_title", "")]
    all_titles += [j.get("title", "") for j in career_history]

    ml_count = sum(1 for t in all_titles if _title_has_ml(t))
    f3 = _normalise(ml_count, 0, 6)   # cap at 6 ML titles

    max_seniority = max((_seniority_of_title(t) for t in all_titles), default=1)
    f4 = max_seniority / 3.0

    current_title = _lower(profile.get("current_title", ""))
    is_coding = any(kw in current_title for kw in CODING_TITLES) and \
                not any(kw in current_title for kw in NON_CODING_TITLES)
    f5 = 1.0 if is_coding else 0.0

    return f3, f4, f5


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 3: Experience (features 6, 7, 8)
# ─────────────────────────────────────────────────────────────────────────────

def experience_features(profile: Dict, career_history: List[Dict]) -> Tuple[float, float, float]:
    """
    f6: total YOE normalised over [0, 15]
    f7: ML-role years normalised over [0, 10]
    f8: Gaussian score centred at 7 years
    """
    yoe = float(profile.get("years_of_experience", 0) or 0)
    f6 = _normalise(yoe, 0, 15)

    ml_months = sum(
        j.get("duration_months", 0) or 0
        for j in career_history
        if _title_has_ml(j.get("title", ""))
    )
    ml_years = ml_months / 12.0
    f7 = _normalise(ml_years, 0, 10)

    f8 = math.exp(-0.5 * ((yoe - YOE_GAUSSIAN_MU) / YOE_GAUSSIAN_SIGMA) ** 2)
    return f6, f7, f8


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 4: Company / Industry (features 9, 10, 11, 12)
# ─────────────────────────────────────────────────────────────────────────────

def _is_consulting(company: str) -> bool:
    c = _lower(company)
    return any(firm in c for firm in CONSULTING_FIRMS)


def _is_product(company: str) -> bool:
    c = _lower(company)
    return any(prod in c for prod in PRODUCT_COMPANIES)


def company_features(profile: Dict, career_history: List[Dict]) -> Tuple[float, float, float, float]:
    """
    f9:  binary – current company is a product company
    f10: binary – ALL companies in career are consulting firms
    f11: binary – ever worked at a FAANG / unicorn (from product list)
    f12: normalised distinct company count (variety of experience)
    """
    current_company = profile.get("current_company", "")
    f9 = 1.0 if _is_product(current_company) else 0.0

    companies = [j.get("company", "") for j in career_history]
    all_consulting = all(_is_consulting(c) for c in companies) if companies else False
    f10 = 1.0 if all_consulting else 0.0

    f11 = 1.0 if any(_is_product(c) for c in companies) else 0.0

    distinct = len({_lower(c) for c in companies})
    f12 = _normalise(distinct, 1, 7)  # 1–7 distinct companies

    return f9, f10, f11, f12


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 5: Education (features 13, 14)
# ─────────────────────────────────────────────────────────────────────────────

def education_features(education: List[Dict]) -> Tuple[float, float]:
    """
    f13: best institution tier score
    f14: binary – any degree in CS/ML/AI/Data-Science
    """
    if not education:
        return 0.3, 0.0  # unknown tier, no CS degree

    best_tier = max(
        EDU_TIER_SCORES.get(e.get("tier", "unknown"), 0.3)
        for e in education
    )
    f13 = best_tier

    has_cs_degree = any(
        any(field_kw in _lower(e.get("field_of_study", "")) for field_kw in CS_ML_FIELDS)
        for e in education
    )
    f14 = 1.0 if has_cs_degree else 0.0
    return f13, f14


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 6: Location (features 15, 16, 17)
# ─────────────────────────────────────────────────────────────────────────────

def location_features(profile: Dict, signals: Dict) -> Tuple[float, float, float]:
    """
    f15: binary – candidate is in India
    f16: binary – willing to relocate
    f17: location bonus: 1.0 Pune/Noida, 0.8 other India, 0.2 outside India
    """
    country = _lower(profile.get("country", ""))
    location = _lower(profile.get("location", ""))
    in_india = country in INDIA_COUNTRY_NAMES or "india" in country
    f15 = 1.0 if in_india else 0.0

    f16 = 1.0 if signals.get("willing_to_relocate", False) else 0.0

    if any(city in location for city in PREFERRED_CITIES):
        f17 = 1.0
    elif in_india:
        f17 = 0.8
    else:
        f17 = 0.2
    return f15, f16, f17


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 7: Red Flags (features 18, 19, 20)
# ─────────────────────────────────────────────────────────────────────────────

def redflag_features(profile: Dict, career_history: List[Dict], skills: List[Dict]) -> Tuple[float, float, float]:
    """
    f18: binary – title-chaser (avg tenure < 18 months)
    f19: binary – all-research / purely academic career
    f20: binary – primary expertise is CV/Speech without NLP/IR exposure
    """
    # f18: Title-chaser check
    tenures = [j.get("duration_months", 0) or 0 for j in career_history]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 0
    f18 = 1.0 if avg_tenure < 18 else 0.0

    # f19: All-research check
    all_titles = [_lower(j.get("title", "")) for j in career_history]
    all_titles.append(_lower(profile.get("current_title", "")))
    is_research = all(
        any(rt in t for rt in RESEARCH_TITLES) or t == ""
        for t in all_titles
    )
    f19 = 1.0 if (is_research and len(all_titles) > 0) else 0.0

    # f20: CV/Speech without NLP/IR
    all_skill_names = {_lower(s.get("name", "")) for s in skills}
    has_cv_speech = any(cv in sn for sn in all_skill_names for cv in CV_SPEECH_ROBOTICS_SKILLS)
    has_nlp_ir = any(nlp in sn for sn in all_skill_names for nlp in NLP_IR_SKILLS)
    f20 = 1.0 if (has_cv_speech and not has_nlp_ir) else 0.0

    return f18, f19, f20


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 8: Behavioral Signals (features 21-27)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def behavioral_features(signals: Dict) -> Tuple[float, ...]:
    """
    Returns 7 behavioral features (all in [0, 1]):
    f21: recruiter_response_rate
    f22: recency_score
    f23: open_to_work_flag
    f24: profile_completeness_score (normalised)
    f25: github_activity_score (normalised; -1 → 0)
    f26: saved_by_recruiters_30d (capped at 20, normalised)
    f27: notice_period_score
    """
    f21 = float(signals.get("recruiter_response_rate", 0) or 0)
    f21 = max(0.0, min(1.0, f21))

    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive <= RECENCY_30:
            f22 = 1.0
        elif days_inactive <= RECENCY_90:
            f22 = 0.6
        else:
            f22 = 0.3
    else:
        f22 = 0.0

    f23 = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    f24 = float(signals.get("profile_completeness_score", 0) or 0) / 100.0
    f24 = max(0.0, min(1.0, f24))

    gh = float(signals.get("github_activity_score", -1) or -1)
    f25 = max(0.0, gh / 100.0) if gh >= 0 else 0.0

    saved = float(signals.get("saved_by_recruiters_30d", 0) or 0)
    f26 = min(saved / 20.0, 1.0)

    notice = int(signals.get("notice_period_days", 90) or 90)
    if notice <= NOTICE_30:
        f27 = 1.0
    elif notice <= NOTICE_60:
        f27 = 0.7
    elif notice <= NOTICE_90:
        f27 = 0.5
    elif notice <= NOTICE_120:
        f27 = 0.3
    else:
        f27 = 0.0

    return f21, f22, f23, f24, f25, f26, f27


# ─────────────────────────────────────────────────────────────────────────────
# Feature Group 9: Honeypot Detection (feature 28)
# ─────────────────────────────────────────────────────────────────────────────

def honeypot_flag(profile: Dict, career_history: List[Dict], skills: List[Dict]) -> float:
    """
    Returns 1.0 if the profile contains a physically impossible signal.
    Uses multiple independent checks to avoid false positives.
    """
    # Check 1: Listed job duration is significantly larger than calendar span
    for job in career_history:
        start_s = job.get("start_date")
        end_s = job.get("end_date")
        dur_listed = job.get("duration_months", 0) or 0
        if start_s:
            try:
                start_dt = datetime.strptime(start_s, "%Y-%m-%d").date()
                if job.get("is_current") or not end_s:
                    end_dt = REFERENCE_DATE
                else:
                    end_dt = datetime.strptime(end_s, "%Y-%m-%d").date()
                cal_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                if dur_listed > cal_months + JOB_DURATION_CALENDAR_SLACK:
                    return 1.0
            except Exception:
                pass

    # Check 2: "Expert" proficiency with 0 duration AND endorsements > 0
    expert_zero_count = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and (s.get("duration_months") or 0) == 0
        and (s.get("endorsements") or 0) > 0
    )
    if expert_zero_count >= EXPERT_ZERO_DUR_THRESHOLD:
        return 1.0

    # Check 3: Endorsements > skill_duration * LIMIT (impossibly high endorsement rate)
    for s in skills:
        ends = s.get("endorsements") or 0
        dur = s.get("duration_months") or 0
        if dur > 0 and ends > dur * ENDORSEMENTS_PER_MONTH_LIMIT:
            return 1.0

    # Check 4: Worked at a company before its documented founding year
    for job in career_history:
        company = _lower(job.get("company", ""))
        start_s = job.get("start_date")
        if start_s:
            try:
                start_year = int(start_s.split("-")[0])
                for comp_kw, founding_year in COMPANY_FOUNDING_YEARS.items():
                    if comp_kw in company and start_year < founding_year - 1:
                        return 1.0
            except Exception:
                pass

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Master extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(candidate: Dict) -> Tuple[List[float], Dict]:
    """
    Extract all 29 features (0-28) from a single candidate record.

    Returns:
        features: list of 29 floats
        meta:     dict with human-readable metadata for reasoning generation
    """
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    edu      = candidate.get("education", [])
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})

    f0, f1, f2 = skill_features(skills)
    f3, f4, f5 = title_features(profile, career)
    f6, f7, f8 = experience_features(profile, career)
    f9, f10, f11, f12 = company_features(profile, career)
    f13, f14 = education_features(edu)
    f15, f16, f17 = location_features(profile, signals)
    f18, f19, f20 = redflag_features(profile, career, skills)
    beh = behavioral_features(signals)
    f21, f22, f23, f24, f25, f26, f27 = beh
    f28 = honeypot_flag(profile, career, skills)

    features = [
        f0, f1, f2,
        f3, f4, f5,
        f6, f7, f8,
        f9, f10, f11, f12,
        f13, f14,
        f15, f16, f17,
        f18, f19, f20,
        f21, f22, f23, f24, f25, f26, f27,
        f28,
    ]

    # ---- Metadata for reasoning ----
    yoe = float(profile.get("years_of_experience", 0) or 0)

    matched_skills = [
        s["name"] for s in skills if _skill_matches_jd(s.get("name", ""))
    ]

    # Skill tier label
    if f0 >= 0.50:
        skill_tier = "Expert"
    elif f0 >= 0.30:
        skill_tier = "Strong"
    elif f0 >= 0.12:
        skill_tier = "Good"
    else:
        skill_tier = "Basic"

    # Behavioral tier label
    beh_score = sum(w * v for w, v in zip(
        [0.20, 0.20, 0.15, 0.10, 0.10, 0.15, 0.10], beh
    ))
    if beh_score >= 0.70:
        beh_tier = "High"
    elif beh_score >= 0.40:
        beh_tier = "Moderate"
    else:
        beh_tier = "Low"

    red_flags_list = []
    if f18:
        red_flags_list.append("title-chaser (short tenures)")
    if f19:
        red_flags_list.append("purely research/academic career")
    if f20:
        red_flags_list.append("CV/Speech focus without NLP/IR")
    if f10:
        red_flags_list.append("all career in consulting firms")
    if f28:
        red_flags_list.append("honeypot / impossible profile")

    meta = {
        "candidate_id": candidate.get("candidate_id", ""),
        "name": profile.get("anonymized_name", ""),
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "years_of_experience": yoe,
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "skill_tier": skill_tier,
        "behavior_tier": beh_tier,
        "matched_skills": matched_skills[:6],   # top 6 for reasoning
        "red_flags": red_flags_list,
        "edu_tier": edu[0].get("tier", "unknown") if edu else "unknown",
        "notice_period_days": int(signals.get("notice_period_days", 90) or 90),
        "open_to_work": bool(signals.get("open_to_work_flag", False)),
        "willing_to_relocate": bool(signals.get("willing_to_relocate", False)),
        "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0) or 0),
        "last_active_date": signals.get("last_active_date", ""),
        "github_score": float(signals.get("github_activity_score", -1) or -1),
        "profile_completeness": float(signals.get("profile_completeness_score", 0) or 0),
        # raw feature values for transparency
        "skill_weighted_score": round(f0, 4),
        "beh_raw_score": round(beh_score, 4),
        "honeypot": bool(f28),
        "all_consulting": bool(f10),
    }

    return features, meta
