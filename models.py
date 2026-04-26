# models.py
# Root of safesignal/ folder — NOT inside environment/
# Step 1 of OpenEnv structure — typed dataclasses
# Shared between server/ and client/ — never import server internals here

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────

class Action(str, Enum):
    """
    The four actions the agent can take each day.
    Core novelty: OBSERVE_QUIETLY has positive reward.
    Teaching an LLM that silence is sometimes optimal
    is the central training challenge.
    """
    OBSERVE_QUIETLY  = "OBSERVE_QUIETLY"
    GENTLE_AWARENESS = "GENTLE_AWARENESS"
    PARENT_CHECK_IN  = "PARENT_CHECK_IN"
    URGENT_SUPPORT   = "URGENT_SUPPORT"


class HiddenRiskState(str, Enum):
    """
    True risk state — never observed by agent (POMDP).
    Agent must infer from behavioral shadows alone.
    This is what makes the problem genuinely hard.
    """
    SAFE       = "SAFE"
    VULNERABLE = "VULNERABLE"
    AT_RISK    = "AT_RISK"
    IN_DANGER  = "IN_DANGER"


class ChildArchetype(str, Enum):
    """
    Three child personality archetypes.
    Agent must identify and adapt strategy per archetype.
    """
    EXPLORER   = "explorer"
    WITHDRAWER = "withdrawer"
    TARGET     = "target"


class GuardianResponse(str, Enum):
    TOOK_ACTION  = "took_action"
    ACKNOWLEDGED = "acknowledged"
    IGNORED      = "ignored"
    NONE         = "none"


# ── Valid Actions List ─────────────────────────────────────────────────────

ACTIONS = [a.value for a in Action]


# ── Observation Dataclass ──────────────────────────────────────────────────

@dataclass
class Observation:
    """
    Everything the agent can observe — 45 behavioral signals.
    Never includes hidden risk state.
    Never includes message content.

    Seven signal clusters address Gaps 3, 6, 7:
        Cluster 1 — Reciprocity
        Cluster 2 — Timing
        Cluster 3 — Migration
        Cluster 4 — Secrecy
        Cluster 5 — Dependency
        Cluster 6 — Social graph
        Cluster 7 — Transactions
    """

    # ── Core behavioral signals ──────────────────────────────────
    activity_hour_mean: float = 20.0
    activity_hour_variance: float = 1.0
    known_contacts_today: int = 5
    unknown_contacts_today: int = 1
    unknown_contact_message_volume: int = 5
    friend_group_engagement_delta: float = 0.0
    family_response_rate: float = 0.75
    sentiment_score: float = 0.5
    sentiment_trend_7d: float = 0.0

    # ── Cluster 1: Reciprocity ────────────────────────────────────
    initiation_ratio: float = 0.5
    message_length_ratio: float = 1.0
    pursuit_score: float = 0.1
    response_time_delta: float = 0.0

    # ── Cluster 2: Timing ─────────────────────────────────────────
    late_night_conversation_rate: float = 0.0
    timing_drift_14d: float = 0.0
    family_avoidance_correlation: float = 0.0
    weekend_intensification: float = 1.0

    # ── Cluster 3: Migration ──────────────────────────────────────
    pre_migration_intensity: float = 0.0
    migration_readiness_score: float = 0.0
    migration_occurred: bool = False
    contact_volume_cliff: bool = False
    cliff_magnitude: float = 0.0
    external_activity_shadow: float = 0.0
    unexplained_late_night_activity: bool = False
    platform_shift_detected: bool = False

    # ── Cluster 4: Secrecy ────────────────────────────────────────
    response_time_variance_by_hour: float = 0.0
    conversation_gap_pattern: float = 0.0
    notification_behavior_change: float = 0.0
    session_interruption_correlation: float = 0.0

    # ── Cluster 5: Dependency ─────────────────────────────────────
    rescue_pattern_score: float = 0.0
    emotional_dependency_score: float = 0.0
    exclusive_confiding_shift: float = 0.0
    contact_sentiment_provision_rate: float = 0.6

    # ── Cluster 6: Social graph ───────────────────────────────────
    active_contact_count_trend: float = 0.0
    single_contact_concentration: float = 0.0
    existing_friendship_decay_rate: float = 0.0
    contact_diversity_score: float = 0.75

    # ── Cluster 7: Transactions ───────────────────────────────────
    received_digital_value: bool = False
    unexplained_account_credits: float = 0.0
    in_app_purchase_pattern_change: float = 0.0

    # ── Guardian context ──────────────────────────────────────────
    guardian_trust: float = 0.8
    days_since_last_alert: int = 999
    last_alert_guardian_response: str = "none"
    consecutive_ignored_alerts: int = 0

    # ── Child profile ─────────────────────────────────────────────
    child_archetype: str = "target"
    day: int = 0

    def to_dict(self) -> dict:
        """Converts to dict for backward compatibility."""
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        """Creates Observation from dict, ignoring unknown keys."""
        valid_fields = {
            k: v for k, v in d.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**valid_fields)


# ── State Dataclass ────────────────────────────────────────────────────────

@dataclass
class State:
    """
    Full internal environment state.
    Includes hidden risk state not visible to agent.
    Used internally by server/environment.py only.
    Client never receives this directly.
    """
    observation: Observation = field(default_factory=Observation)
    hidden_risk_state: HiddenRiskState = HiddenRiskState.SAFE
    hidden_risk_numeric: int = 0
    day: int = 0
    episode_done: bool = False
    guardian_trust: float = 0.8
    consecutive_ignored_alerts: int = 0
    migration_occurred: bool = False
    migration_day: Optional[int] = None


# ── Step Result Dataclass ──────────────────────────────────────────────────

@dataclass
class StepResult:
    """
    Returned by environment.step().
    Follows standard Gym-style interface.
    rubric_scores shows composable breakdown —
    judges can see exactly how reward was computed.
    """
    observation: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)
    rubric_scores: dict = field(default_factory=dict)


# ── Reset Result Dataclass ─────────────────────────────────────────────────

@dataclass
class ResetResult:
    """Returned by environment.reset()."""
    observation: Observation
    info: dict = field(default_factory=dict)


# ── Episode Summary Dataclass ──────────────────────────────────────────────

@dataclass
class EpisodeSummary:
    """
    Full episode summary for plotting and evaluation.
    Person B uses this for reward curves.
    Person C uses this for demo visualization.
    """
    total_reward: float
    final_hidden_state: str
    final_guardian_trust: float
    total_interventions: int
    rubric_averages: dict = field(default_factory=dict)
    trust_trajectory: list = field(default_factory=list)
    hidden_trajectory: list = field(default_factory=list)
    reward_trajectory: list = field(default_factory=list)


# ── Quick Verification ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Verifying models.py...\n")

    # Test Action enum
    print(f"Actions: {[a.value for a in Action]}")

    # Test Observation creation
    obs = Observation()
    print(f"Default observation keys: {len(obs.to_dict())}")

    # Test from_dict
    d = {"guardian_trust": 0.6, "child_archetype": "withdrawer",
         "unknown_key": "ignored"}
    obs2 = Observation.from_dict(d)
    print(f"from_dict trust: {obs2.guardian_trust}")
    print(f"from_dict archetype: {obs2.child_archetype}")

    # Test State
    state = State()
    print(f"Default hidden state: {state.hidden_risk_state}")

    # Test StepResult
    result = StepResult(
        observation=obs,
        reward=0.5,
        done=False,
        rubric_scores={"intervention_timing": {"weighted_score": 0.2}}
    )
    print(f"StepResult reward: {result.reward}")

    print("\n✅ models.py working correctly")
    print(f"ACTIONS list: {ACTIONS}")