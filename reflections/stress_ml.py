"""
stress_ml.py  —  Stress Coach ML prediction module
=====================================================
Uses a Random-Forest-based pipeline trained on synthetic data.
No internet required; scikit-learn only.

Public API
----------
    from reflections.stress_ml import predict_weekly_reflection

    result = predict_weekly_reflection({
        "avg_stress_level":      42,    # 0-100
        "avg_study_hours":       5.5,   # hours/day
        "dominant_confidence":   "low", # low / medium / high
        "total_deadlines":       3,
        "completed_checkins":    5,     # out of 7
        "user_summary":          "Felt overwhelmed this week...",
    })

Returns a dict with:
    burnout_risk         – "low" | "moderate" | "high"
    burnout_score        – 0.0–1.0
    stress_trend         – "improving" | "stable" | "worsening"
    recommendation       – plain-English coaching paragraph
    focus_areas          – list[str]  (top 3 action items)
    predicted_next_stress– float (predicted stress level next week)
    wellness_score       – float 0-100
"""

import re
import random
import hashlib
import numpy as np

# ── optional scikit-learn (graceful fallback if not installed) ─────────────
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ════════════════════════════════════════════════════════════════════════════
# 1.  Synthetic training data generator
# ════════════════════════════════════════════════════════════════════════════

def _generate_synthetic_data(n=800, seed=42):
    """Return (X, y_burnout, y_stress_next) numpy arrays."""
    rng = np.random.RandomState(seed)

    stress        = rng.uniform(0, 100, n)
    study_hours   = rng.uniform(0, 12, n)
    confidence    = rng.choice([0, 1, 2], n)   # 0=low 1=medium 2=high
    deadlines     = rng.randint(0, 10, n)
    checkins      = rng.randint(0, 8, n)
    checkin_rate  = np.clip(checkins / 7, 0, 1)

    # Burnout risk: deterministic rule + noise
    burnout_score = (
        0.40 * stress / 100
        + 0.20 * np.clip(study_hours / 12, 0, 1)
        + 0.15 * (1 - confidence / 2)
        + 0.15 * np.clip(deadlines / 10, 0, 1)
        - 0.10 * checkin_rate
        + rng.normal(0, 0.05, n)
    )
    burnout_score = np.clip(burnout_score, 0, 1)

    burnout_label = np.where(
        burnout_score < 0.35, 0,
        np.where(burnout_score < 0.60, 1, 2)
    )  # 0=low 1=moderate 2=high

    # Next-week stress prediction
    next_stress = (
        0.55 * stress
        + 0.12 * study_hours * 5
        + 0.10 * (2 - confidence) * 15
        + 0.08 * deadlines * 3
        - 0.05 * checkin_rate * 20
        + rng.normal(0, 5, n)
    )
    next_stress = np.clip(next_stress, 0, 100)

    X = np.column_stack([
        stress, study_hours, confidence, deadlines, checkin_rate
    ])
    return X, burnout_label, next_stress


# ════════════════════════════════════════════════════════════════════════════
# 2.  Train once at import time
# ════════════════════════════════════════════════════════════════════════════

_BURNOUT_LABELS = ["low", "moderate", "high"]
_CONFIDENCE_MAP = {"low": 0, "medium": 1, "high": 2}

if SKLEARN_OK:
    _X, _y_burn, _y_stress = _generate_synthetic_data()
    _clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    _clf.fit(_X, _y_burn)
    _reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    _reg.fit(_X, _y_stress)
else:
    _clf = _reg = None


# ════════════════════════════════════════════════════════════════════════════
# 3.  Text-based sentiment helper (no NLTK needed)
# ════════════════════════════════════════════════════════════════════════════

_NEGATIVE_WORDS = {
    "overwhelm", "exhaust", "stress", "anxious", "anxiety", "tired",
    "burnout", "difficult", "struggle", "fail", "lost", "hopeless",
    "frustrated", "panic", "overload", "depressed", "unmotivated",
    "distracted", "worry", "worried", "bad", "terrible", "awful",
}
_POSITIVE_WORDS = {
    "good", "great", "calm", "relax", "balance", "happy", "confident",
    "progress", "achieve", "success", "motivated", "focus", "productive",
    "energetic", "excellent", "improve", "better", "manage", "control",
}


def _text_sentiment(text: str) -> float:
    """Return a sentiment score: -1.0 (very negative) … +1.0 (very positive)."""
    if not text:
        return 0.0
    words = set(re.findall(r"[a-z]+", text.lower()))
    neg = len(words & _NEGATIVE_WORDS)
    pos = len(words & _POSITIVE_WORDS)
    total = neg + pos or 1
    return (pos - neg) / total


# ════════════════════════════════════════════════════════════════════════════
# 4.  Deterministic fallback (no sklearn)
# ════════════════════════════════════════════════════════════════════════════

def _rule_based_predict(stress, study_hours, confidence_code,
                        deadlines, checkin_rate, sentiment):
    score = (
        0.40 * stress / 100
        + 0.18 * min(study_hours / 12, 1)
        + 0.15 * (1 - confidence_code / 2)
        + 0.14 * min(deadlines / 10, 1)
        - 0.10 * checkin_rate
        - 0.03 * max(sentiment, 0)
    )
    score = max(0.0, min(1.0, score))

    if score < 0.35:
        label = "low"
    elif score < 0.60:
        label = "moderate"
    else:
        label = "high"

    next_s = (
        0.55 * stress
        + 0.12 * study_hours * 5
        + 0.10 * (2 - confidence_code) * 15
        + 0.08 * deadlines * 3
        - 0.05 * checkin_rate * 20
    )
    next_s = max(0.0, min(100.0, next_s))
    return label, score, next_s


# ════════════════════════════════════════════════════════════════════════════
# 5.  Recommendation engine
# ════════════════════════════════════════════════════════════════════════════

_RECOMMENDATIONS = {
    "low": (
        "Great job managing your stress this week! "
        "Your check-in consistency is paying off. "
        "Keep your current study routine and remember to schedule short recovery breaks. "
        "Consider setting a slightly more ambitious goal for next week while protecting sleep.",
        ["Maintain current study schedule", "Add 10-min mindfulness sessions", "Plan one leisure activity"],
    ),
    "moderate": (
        "You're carrying a moderate load right now — that's manageable with the right adjustments. "
        "Try breaking large tasks into 25-minute Pomodoro blocks and tackle the hardest deadline first each day. "
        "Prioritise 7–8 hours of sleep; it directly lowers perceived stress. "
        "A short walk or breathing exercise between study sessions can reset your focus.",
        ["Use Pomodoro technique for study blocks", "Protect 7–8 h of sleep nightly", "30-min outdoor walk each day"],
    ),
    "high": (
        "Your indicators suggest a high risk of burnout — please take this seriously. "
        "Start by reducing non-essential commitments and communicating deadline pressures to your instructors or supervisor. "
        "Schedule at least one full rest day this week with no academic work. "
        "Daily 5-minute breathing exercises (box breathing: 4-4-4-4) can quickly lower cortisol. "
        "Consider speaking with a counsellor or mentor — you don't need to navigate this alone.",
        ["Request deadline extension where possible", "Take one full rest day this week", "Practice box-breathing 3× daily"],
    ),
}


def _stress_trend(stress, checkin_rate, sentiment):
    """Heuristic trend based on available signals."""
    score = checkin_rate * 0.5 + max(sentiment, 0) * 0.3 + (1 - stress / 100) * 0.2
    if score > 0.55:
        return "improving"
    elif score > 0.35:
        return "stable"
    else:
        return "worsening"


def _wellness_score(stress, study_hours, checkin_rate, confidence_code, sentiment):
    """0-100 wellness score, higher = better."""
    s = (
        (1 - stress / 100) * 40
        + min(study_hours / 8, 1) * 20   # sweet-spot ~8 h study
        + checkin_rate * 20
        + (confidence_code / 2) * 10
        + max(sentiment, 0) * 10
    )
    return round(max(0.0, min(100.0, s)), 1)


# ════════════════════════════════════════════════════════════════════════════
# 6.  Public API
# ════════════════════════════════════════════════════════════════════════════

def predict_weekly_reflection(data: dict) -> dict:
    """
    Parameters
    ----------
    data : dict
        avg_stress_level      float  0-100
        avg_study_hours       float  hours/day
        dominant_confidence   str    "low" | "medium" | "high"
        total_deadlines       int
        completed_checkins    int    (out of 7)
        user_summary          str    (optional free text)

    Returns
    -------
    dict with keys:
        burnout_risk, burnout_score, stress_trend,
        recommendation, focus_areas,
        predicted_next_stress, wellness_score
    """
    stress          = float(data.get("avg_stress_level", 50))
    study_hours     = float(data.get("avg_study_hours", 4))
    confidence_str  = str(data.get("dominant_confidence", "medium")).lower()
    confidence_code = _CONFIDENCE_MAP.get(confidence_str, 1)
    deadlines       = int(data.get("total_deadlines", 0))
    checkins        = int(data.get("completed_checkins", 0))
    checkin_rate    = min(checkins / 7, 1.0)
    summary_text    = str(data.get("user_summary", ""))

    sentiment = _text_sentiment(summary_text)

    if SKLEARN_OK and _clf is not None:
        X = np.array([[stress, study_hours, confidence_code, deadlines, checkin_rate]])
        burn_idx     = int(_clf.predict(X)[0])
        burn_proba   = float(_clf.predict_proba(X)[0][burn_idx])
        burnout_risk = _BURNOUT_LABELS[burn_idx]
        # blend proba with rule-based score for smoothness
        _, rule_score, _ = _rule_based_predict(
            stress, study_hours, confidence_code, deadlines, checkin_rate, sentiment
        )
        burnout_score = round(0.6 * burn_proba + 0.4 * rule_score, 3)
        next_stress   = round(float(_reg.predict(X)[0]), 1)
    else:
        burnout_risk, burnout_score, next_stress = _rule_based_predict(
            stress, study_hours, confidence_code, deadlines, checkin_rate, sentiment
        )
        burnout_score = round(burnout_score, 3)
        next_stress   = round(next_stress, 1)

    # Adjust for text sentiment
    if sentiment < -0.3 and burnout_risk == "moderate":
        burnout_risk = "high"
    elif sentiment > 0.3 and burnout_risk == "moderate":
        burnout_risk = "low"

    recommendation, focus_areas = _RECOMMENDATIONS[burnout_risk]
    trend    = _stress_trend(stress, checkin_rate, sentiment)
    wellness = _wellness_score(stress, study_hours, checkin_rate, confidence_code, sentiment)

    return {
        "burnout_risk":          burnout_risk,
        "burnout_score":         burnout_score,
        "stress_trend":          trend,
        "recommendation":        recommendation,
        "focus_areas":           focus_areas,
        "predicted_next_stress": next_stress,
        "wellness_score":        wellness,
    }