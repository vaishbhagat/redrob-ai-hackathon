"""
config.py — Central configuration for Redrob Candidate Ranking System
All constants, keyword lists, weights, and thresholds live here.
"""

from datetime import date

# ---------------------------------------------------------------------------
# Reference date (used for recency scoring and calendar-duration checks)
# ---------------------------------------------------------------------------
REFERENCE_DATE = date(2026, 6, 22)

# ---------------------------------------------------------------------------
# JD Skill Keywords (exact and partial match; all lowercase)
# ---------------------------------------------------------------------------
JD_SKILLS = [
    "embeddings", "retrieval", "ranking", "vector database",
    "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss",
    "llm", "fine-tuning", "lora", "peft",
    "python", "ndcg", "mrr", "map", "learning-to-rank",
    "xgboost", "recommendation", "search", "information retrieval",
    "transformers", "bert", "sentence transformers", "bge", "e5",
    # Partial / alias variants that appear in profiles
    "sentence transformer", "hugging face", "huggingface",
    "semantic search", "bm25", "hybrid search", "re-ranking",
    "lora", "qlora", "peft", "fine tuning", "finetuning",
    "learning to rank", "letor", "ranknet", "lambdamart",
    "vector search", "ann", "hnsw",
]

# ---------------------------------------------------------------------------
# Title keywords for ML-role detection
# ---------------------------------------------------------------------------
ML_TITLE_KEYWORDS = [
    "ml", "machine learning", "ai", "artificial intelligence",
    "data scientist", "data science",
    "recommendation", "search", "ranking", "retrieval",
    "nlp", "natural language", "applied scientist",
    "research scientist",          # included for scoring, but penalised elsewhere
    "computer vision",             # adjacent; lower weight implicitly
    "deep learning",
]

# Keywords that indicate seniority levels (checked in order; first match wins)
SENIORITY_LEVELS = {
    3: ["principal", "staff", "director", "vp ", "head of", "lead", "manager"],
    2: ["senior", "sr.", "sr "],
    1: ["engineer", "scientist", "developer", "analyst", "architect"],
    0: ["intern", "trainee", "associate", "junior", "jr.", "jr "],
}

# Titles that imply the person is actually writing code (feature 6)
CODING_TITLES = [
    "engineer", "developer", "scientist", "architect", "analyst",
    "sde", "mle", "ml engineer", "software", "data",
]

NON_CODING_TITLES = [
    "intern", "trainee", "associate (entry)", "hr ", "recruiter",
    "accountant", "marketing", "operations manager", "support",
    "content writer", "graphic", "civil engineer", "mechanical engineer",
]

# ---------------------------------------------------------------------------
# Company classification lists (all lowercase)
# ---------------------------------------------------------------------------
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "hcl", "tech mahindra", "lti",
    "mphasis", "hexaware", "zensar", "niit technologies",
    "cyient", "mastech", "sonata software", "ltimindtree",
    "l&t infotech", "persistent systems",
}

PRODUCT_COMPANIES = {
    "swiggy", "zomato", "uber", "ola", "flipkart",
    "amazon", "google", "microsoft", "razorpay", "cred",
    "paytm", "byju", "upstox", "groww", "meesho",
    "phonepe", "freshworks", "zoho", "atlassian", "adobe",
    "netflix", "meta", "apple", "salesforce", "linkedin",
    "nykaa", "dream11", "slice", "sharechat",
    # fictitonous but product-company archetypes in the synthetic data
    "pied piper", "initech", "globex", "stark industries", "wayne enterprises",
    "hooli",
}

# Companies NOT to penalise even though they look like consulting from name
PRODUCT_CONSULTING_EXCEPTIONS = {
    "dunder mifflin",  # clearly fictional
}

# ---------------------------------------------------------------------------
# Education tier scores
# ---------------------------------------------------------------------------
EDU_TIER_SCORES = {
    "tier_1": 1.0,
    "tier_2": 0.7,
    "tier_3": 0.4,
    "tier_4": 0.2,
    "unknown": 0.3,
}

# Fields of study that count as CS/ML relevant (feature 15)
CS_ML_FIELDS = [
    "computer science", "cs", "machine learning", "artificial intelligence",
    "data science", "information technology", "software engineering",
    "electronics", "electrical", "statistics", "mathematics",
    "information systems", "computational", "it",
]

# ---------------------------------------------------------------------------
# Location mappings
# ---------------------------------------------------------------------------
PREFERRED_CITIES = {"pune", "noida"}
INDIA_COUNTRY_NAMES = {"india"}

# ---------------------------------------------------------------------------
# Red-flag patterns
# ---------------------------------------------------------------------------
# Feature 20: purely research / academic titles
RESEARCH_TITLES = [
    "research scientist", "postdoc", "postdoctoral", "researcher",
    "research associate", "research engineer", "phd student",
    "research fellow", "academic",
]

# Feature 21: primary skills that disqualify if NLP/IR not present
CV_SPEECH_ROBOTICS_SKILLS = {
    "computer vision", "object detection", "image classification",
    "speech recognition", "speech synthesis", "tts", "asr",
    "robotics", "ros", "slam", "image segmentation", "yolo",
    "opencv", "mediapipe", "gans", "diffusion models",
}

NLP_IR_SKILLS = {
    "nlp", "natural language", "information retrieval", "llm",
    "embeddings", "bert", "transformers", "search", "ranking",
    "recommendation", "rag", "semantic search",
}

# ---------------------------------------------------------------------------
# Proficiency weights for skill matching
# ---------------------------------------------------------------------------
PROFICIENCY_WEIGHTS = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.8,
    "expert": 1.0,
}

# Cap on skill duration for normalisation (months)
SKILL_DURATION_CAP = 36

# ---------------------------------------------------------------------------
# Experience Gaussian target (years)
# ---------------------------------------------------------------------------
YOE_GAUSSIAN_MU = 7.0   # ideal centre
YOE_GAUSSIAN_SIGMA = 2.0

# ---------------------------------------------------------------------------
# Feature weight vector (28 features, 0-indexed)
#
# Features 0-2:   Skill match (3)
# Features 3-5:   Title fit (3)
# Features 6-8:   Experience (3)
# Features 9-12:  Company / industry (4)
# Features 13-14: Education (2)
# Features 15-17: Location (3)
# Features 18-20: Red flags — PENALTIES (not in base score)
# Features 21-27: Behavioral signals (7)
# Feature  28:    Honeypot flag — sets score to -inf
#
# The weight vector below applies to features 0-17 only (the positive features).
# Red-flag and honeypot features are applied as multipliers/overrides separately.
# ---------------------------------------------------------------------------

# Individual weights (sum ≈ 1.0 for interpretability)
FEATURE_WEIGHTS = [
    # --- Skill (3) ---  total = 0.38
    0.22,   # 0: weighted skill overlap score
    0.08,   # 1: avg proficiency of matched skills
    0.08,   # 2: avg duration of matched skills (normalised)

    # --- Title (3) ---  total = 0.15
    0.07,   # 3: count of ML-relevant past/current titles
    0.05,   # 4: max seniority level
    0.03,   # 5: binary: is in a coding/technical role

    # --- Experience (3) ---  total = 0.17
    0.04,   # 6: total years of experience (scaled 0-1 over 0-15 yrs)
    0.08,   # 7: years in ML roles (scaled)
    0.05,   # 8: Gaussian score centred at 7 yrs

    # --- Company / industry (4) ---  total = 0.10
    0.04,   # 9:  binary: current company is product company
    0.00,   # 10: binary: ALL companies are consulting (penalty only; weight 0)
    0.04,   # 11: binary: ever worked at FAANG/unicorn
    0.02,   # 12: distinct company count (normalised; more = broader exposure)

    # --- Education (2) ---  total = 0.05
    0.03,   # 13: institution tier score
    0.02,   # 14: binary: CS/ML/AI degree

    # --- Location (3) ---  total = 0.05
    0.01,   # 15: binary: located in India
    0.01,   # 16: binary: willing to relocate
    0.03,   # 17: location bonus (Pune/Noida=1.0, other India=0.8, outside=0.2)

    # --- Red flags (3) --- PENALTIES applied separately
    0.00,   # 18: title-chaser flag (avg tenure < 18 months)
    0.00,   # 19: all-research flag
    0.00,   # 20: CV/Speech without NLP/IR

    # --- Behavioral (7) --- used in behavioral modifier, NOT in base score
    0.00,   # 21: recruiter_response_rate
    0.00,   # 22: recency_score
    0.00,   # 23: open_to_work_flag
    0.00,   # 24: profile_completeness_score
    0.00,   # 25: github_activity_score
    0.00,   # 26: saved_by_recruiters_30d (normalised)
    0.00,   # 27: notice_period_score
]

# ---------------------------------------------------------------------------
# Behavioral modifier weights (features 21-27)
# Weighted sum → scaled to 0.40–1.50 multiplier
# ---------------------------------------------------------------------------
BEHAVIORAL_WEIGHTS = [
    0.20,   # recruiter_response_rate
    0.20,   # recency_score
    0.15,   # open_to_work_flag
    0.10,   # profile_completeness_score
    0.10,   # github_activity_score
    0.15,   # saved_by_recruiters_30d
    0.10,   # notice_period_score
]

# Behavioral modifier scaling
BEH_MOD_MIN = 0.40
BEH_MOD_MAX = 1.50

# ---------------------------------------------------------------------------
# Red-flag penalty multipliers (applied multiplicatively)
# ---------------------------------------------------------------------------
TITLE_CHASER_PENALTY   = 0.65   # avg tenure < 18 months
ALL_RESEARCH_PENALTY   = 0.50   # purely academic career
CV_ONLY_PENALTY        = 0.60   # CV/speech without NLP/IR
ALL_CONSULTING_PENALTY = 0.80   # entire career in consulting firms

# ---------------------------------------------------------------------------
# Honeypot detection thresholds
# ---------------------------------------------------------------------------
# Job duration listed in profile vs calendar duration between start/end date
JOB_DURATION_CALENDAR_SLACK = 12  # months tolerance

# Expert skill with 0 duration months AND endorsements > 0
EXPERT_ZERO_DUR_THRESHOLD = 1  # at least this many such skills → honeypot

# Endorsements per month threshold
ENDORSEMENTS_PER_MONTH_LIMIT = 10  # endorsements > dur * this → suspicious

# Calendar check: job start date before company founding year
COMPANY_FOUNDING_YEARS = {
    "cred": 2018,
    "groww": 2016,
    "swiggy": 2014,
    "razorpay": 2014,
    "byju": 2011,
    "paytm": 2010,
    "ola": 2010,
    "upstox": 2009,
    "zomato": 2008,
    "flipkart": 2007,
}

# ---------------------------------------------------------------------------
# Notice period score thresholds (days)
# ---------------------------------------------------------------------------
NOTICE_30  = 30
NOTICE_60  = 60
NOTICE_90  = 90
NOTICE_120 = 120

# ---------------------------------------------------------------------------
# Recency score thresholds (days from REFERENCE_DATE)
# ---------------------------------------------------------------------------
RECENCY_30 = 30
RECENCY_90 = 90

# ---------------------------------------------------------------------------
# Feature names (for documentation / debugging)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "skill_weighted_score",        # 0
    "skill_avg_proficiency",       # 1
    "skill_avg_duration",          # 2
    "title_ml_count",              # 3
    "title_seniority",             # 4
    "title_is_coding",             # 5
    "yoe_total",                   # 6
    "yoe_ml_roles",                # 7
    "yoe_gaussian",                # 8
    "company_is_product",          # 9
    "company_all_consulting",      # 10
    "company_faang_unicorn",       # 11
    "company_distinct_count",      # 12
    "edu_tier_score",              # 13
    "edu_cs_ml_degree",            # 14
    "location_india",              # 15
    "location_willing_relocate",   # 16
    "location_bonus",              # 17
    "redflag_title_chaser",        # 18
    "redflag_all_research",        # 19
    "redflag_cv_only",             # 20
    "beh_recruiter_response",      # 21
    "beh_recency",                 # 22
    "beh_open_to_work",            # 23
    "beh_profile_completeness",    # 24
    "beh_github_activity",         # 25
    "beh_saved_recruiters",        # 26
    "beh_notice_period",           # 27
    "honeypot_flag",               # 28
]
