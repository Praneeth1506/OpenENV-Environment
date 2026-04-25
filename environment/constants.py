# constants.py
# All simulation parameters calibrated to published research
# Sources: Thorn.org/research, CCRC UNH, Pew Research teen usage studies

# ── Archetype Baseline Behaviors ──────────────────────────────────────────

EXPLORER_BASELINE = {
    "activity_hour_mean": 20.0,       # active around 8pm normally
    "activity_hour_std": 2.5,         # high variance — normal for them
    "known_contacts_daily": 8,        # meets many people
    "unknown_contacts_daily": 3,      # new contacts normal
    "friend_engagement_baseline": 0.8,
    "family_response_baseline": 0.7,
    "sentiment_baseline": 0.6,
}

WITHDRAWER_BASELINE = {
    "activity_hour_mean": 19.0,       # active around 7pm normally
    "activity_hour_std": 0.8,         # low variance — creature of habit
    "known_contacts_daily": 3,        # small close group
    "unknown_contacts_daily": 0.5,    # rarely meets new people
    "friend_engagement_baseline": 0.6,
    "family_response_baseline": 0.8,  # closer to family
    "sentiment_baseline": 0.5,
}

TARGET_BASELINE = {
    "activity_hour_mean": 19.5,       # normal timing initially
    "activity_hour_std": 1.0,
    "known_contacts_daily": 6,        # social but not extreme
    "unknown_contacts_daily": 1,      # low unknown contact baseline
    "friend_engagement_baseline": 0.75,
    "family_response_baseline": 0.75,
    "sentiment_baseline": 0.65,
}

# ── Risk State Transition Probabilities ───────────────────────────────────
# Based on Thorn research: average grooming timeline 3-6 weeks
# These probabilities govern hidden state changes per day

TRANSITION_PROBS = {
    # Format: (current_state, condition) → probability of advancing
    "safe_to_vulnerable": 0.05,       # baseline daily drift probability
    "vulnerable_to_at_risk": 0.08,
    "at_risk_to_in_danger": 0.10,
    # Recovery probabilities (when guardian takes action)
    "recovery_after_action": 0.65,    # good conversation → risk reduces
    "self_recovery": 0.10,            # child resolves on their own
}

# ── Guardian Model Parameters ─────────────────────────────────────────────

GUARDIAN_INITIAL_TRUST = 0.8
GUARDIAN_TRUST_DECAY_FALSE_ALARM = 0.15
GUARDIAN_TRUST_DECAY_IGNORED = 0.05
GUARDIAN_TRUST_DECAY_TOO_FREQUENT = 0.08
GUARDIAN_TRUST_RECOVERY_PER_QUIET_DAY = 0.03
GUARDIAN_TRUST_BOOST_ACTION_TAKEN = 0.10
GUARDIAN_ALERT_FATIGUE_THRESHOLD = 3   # ignored alerts before tuning out

# ── Episode Parameters ────────────────────────────────────────────────────

EPISODE_LENGTH = 30                   # days — captures full grooming arc
MIN_DAYS_BETWEEN_ALERTS = 2          # minimum gap before alerting again