# simulated_child.py
import random
import math
import numpy as np
from constants import *

class SimulatedChild:
    def __init__(self, archetype=None):
        # Randomly select archetype if not specified
        if archetype is None:
            archetype = random.choice(["explorer", "withdrawer", "target"])
        
        self.archetype = archetype
        self.baseline = self._get_baseline()
        self.day = 0
        
        # Hidden risk state — environment sees this, agent never does
        self.hidden_state = "SAFE"
        self.hidden_state_numeric = 0  # 0=SAFE 1=VULNERABLE 2=AT_RISK 3=IN_DANGER
        
        # Behavioral tracking
        self.consecutive_drift_days = 0
        self.consecutive_aligned_days = 0
        self.unknown_contact_relationship_days = 0
        
        # Guardian state
        self.guardian_trust = GUARDIAN_INITIAL_TRUST
        self.days_since_last_alert = 999  # no alert sent yet
        self.last_alert_response = "none"
        self.consecutive_ignored_alerts = 0
        
    def _get_baseline(self):
        if self.archetype == "explorer":
            return EXPLORER_BASELINE
        elif self.archetype == "withdrawer":
            return WITHDRAWER_BASELINE
        elif self.archetype == "target":
            return TARGET_BASELINE

    def get_observable_state(self):
        """Returns the state vector the agent observes. Never includes hidden_state."""
        return {
            "activity_hour_mean": self._get_activity_hour(),
            "activity_hour_variance": self._get_activity_variance(),
            "known_contacts_today": self._get_known_contacts(),
            "unknown_contacts_today": self._get_unknown_contacts(),
            "unknown_contact_message_volume": self._get_unknown_volume(),
            "friend_group_engagement_delta": self._get_friend_delta(),
            "family_response_rate": self._get_family_response(),
            "sentiment_score": self._get_sentiment(),
            "sentiment_trend_7d": self._get_sentiment_trend(),
            "days_since_last_alert": self.days_since_last_alert,
            "last_alert_guardian_response": self.last_alert_response,
            "guardian_trust": self.guardian_trust,
            "consecutive_ignored_alerts": self.consecutive_ignored_alerts,
            "child_archetype": self.archetype,
        }

    def _get_activity_hour(self):
        """Activity hour shifts later as risk increases."""
        base = self.baseline["activity_hour_mean"]
        risk_shift = self.hidden_state_numeric * 1.2  # shifts up to ~3.6hrs later
        noise = random.gauss(0, self.baseline["activity_hour_std"])
        return min(base + risk_shift + noise, 26.0)  # cap at 2am

    def _get_activity_variance(self):
        base_std = self.baseline["activity_hour_std"]
        risk_increase = self.hidden_state_numeric * 0.5
        return base_std + risk_increase

    def _get_known_contacts(self):
        """Known contacts decrease as child is isolated."""
        base = self.baseline["known_contacts_daily"]
        isolation_factor = 1.0 - (self.hidden_state_numeric * 0.2)
        noise = random.randint(-1, 1)
        return max(0, int(base * isolation_factor + noise))

    def _get_unknown_contacts(self):
        """Unknown contacts increase as grooming progresses."""
        base = self.baseline["unknown_contacts_daily"]
        if self.hidden_state in ["AT_RISK", "IN_DANGER"]:
            return int(base + self.hidden_state_numeric * 2 + random.randint(0, 2))
        return max(0, int(base + random.randint(-1, 1)))

    def _get_unknown_volume(self):
        """Message volume with unknown contacts — key grooming signal."""
        if self.hidden_state == "SAFE":
            return random.randint(0, 5)
        elif self.hidden_state == "VULNERABLE":
            return random.randint(5, 20)
        elif self.hidden_state == "AT_RISK":
            return random.randint(20, 60)
        elif self.hidden_state == "IN_DANGER":
            return random.randint(50, 120)

    def _get_friend_delta(self):
        """
        Change in friend group engagement vs personal baseline.
        Negative means declining. Key isolation signal.
        """
        base_engagement = self.baseline["friend_engagement_baseline"]
        risk_decay = self.hidden_state_numeric * 0.18
        current = base_engagement - risk_decay + random.gauss(0, 0.05)
        return round(current - base_engagement, 3)

    def _get_family_response(self):
        """Family response rate drops as child becomes isolated."""
        base = self.baseline["family_response_baseline"]
        decay = self.hidden_state_numeric * 0.15
        noise = random.gauss(0, 0.05)
        return round(max(0.0, min(1.0, base - decay + noise)), 3)

    def _get_sentiment(self):
        """Emotional tone declines with risk level."""
        base = self.baseline["sentiment_baseline"]
        decline = self.hidden_state_numeric * 0.2
        noise = random.gauss(0, 0.08)
        return round(max(-1.0, min(1.0, base - decline + noise)), 3)

    def _get_sentiment_trend(self):
        """7-day trend. Negative = declining over past week."""
        if self.hidden_state == "SAFE":
            return round(random.gauss(0.0, 0.05), 3)
        elif self.hidden_state == "VULNERABLE":
            return round(random.gauss(-0.1, 0.05), 3)
        elif self.hidden_state == "AT_RISK":
            return round(random.gauss(-0.25, 0.05), 3)
        elif self.hidden_state == "IN_DANGER":
            return round(random.gauss(-0.4, 0.05), 3)

    def advance_day(self, agent_action, guardian_response):
        """
        Called by the environment's step() function.
        Updates hidden state, guardian trust, and day counter.
        Returns whether the hidden state changed.
        """
        self.day += 1
        self.days_since_last_alert += 1

        # Update guardian trust based on what happened
        self._update_guardian_trust(agent_action, guardian_response)

        # Store previous hidden state to detect changes
        prev_hidden = self.hidden_state

        # Transition hidden risk state
        self._transition_hidden_state(agent_action, guardian_response)

        return self.hidden_state != prev_hidden

    def _update_guardian_trust(self, action, guardian_response):
        if action == "OBSERVE_QUIETLY":
            # Quiet days slowly rebuild trust
            self.guardian_trust = min(
                1.0,
                self.guardian_trust + GUARDIAN_TRUST_RECOVERY_PER_QUIET_DAY
            )
            self.days_since_last_alert += 1
        else:
            # An alert was sent
            self.days_since_last_alert = 0

            if guardian_response == "took_action":
                self.guardian_trust = min(
                    1.0,
                    self.guardian_trust + GUARDIAN_TRUST_BOOST_ACTION_TAKEN
                )
                self.consecutive_ignored_alerts = 0
            elif guardian_response == "acknowledged":
                pass  # neutral, no trust change
            elif guardian_response == "ignored":
                self.guardian_trust = max(
                    0.0,
                    self.guardian_trust - GUARDIAN_TRUST_DECAY_IGNORED
                )
                self.consecutive_ignored_alerts += 1

            # Too frequent penalty
            if self.days_since_last_alert < MIN_DAYS_BETWEEN_ALERTS:
                self.guardian_trust = max(
                    0.0,
                    self.guardian_trust - GUARDIAN_TRUST_DECAY_TOO_FREQUENT
                )

        self.last_alert_response = guardian_response

    def _transition_hidden_state(self, action, guardian_response):
        """
        Probabilistic hidden state transitions.
        Guardian action can push state back toward safe.
        """
        # Recovery path — guardian took action
        if guardian_response == "took_action":
            if self.hidden_state_numeric > 0:
                if random.random() < TRANSITION_PROBS["recovery_after_action"]:
                    self.hidden_state_numeric = max(
                        0, self.hidden_state_numeric - 1
                    )

        # Self recovery — child resolves on own
        elif action == "OBSERVE_QUIETLY" and self.hidden_state_numeric > 0:
            if random.random() < TRANSITION_PROBS["self_recovery"]:
                self.hidden_state_numeric = max(
                    0, self.hidden_state_numeric - 1
                )

        # Progression — risk increasing
        else:
            prob_map = {
                0: TRANSITION_PROBS["safe_to_vulnerable"],
                1: TRANSITION_PROBS["vulnerable_to_at_risk"],
                2: TRANSITION_PROBS["at_risk_to_in_danger"],
                3: 0.0  # already at maximum risk
            }
            if random.random() < prob_map[self.hidden_state_numeric]:
                self.hidden_state_numeric = min(
                    3, self.hidden_state_numeric + 1
                )

        # Sync string representation
        state_map = {0: "SAFE", 1: "VULNERABLE", 2: "AT_RISK", 3: "IN_DANGER"}
        self.hidden_state = state_map[self.hidden_state_numeric]

    def simulate_guardian_response(self, action):
        """
        Simulates how the guardian responds to an alert.
        Based on current trust level and alert fatigue.
        """
        if action == "OBSERVE_QUIETLY":
            return "none"

        # High fatigue — likely to ignore
        if self.consecutive_ignored_alerts >= GUARDIAN_ALERT_FATIGUE_THRESHOLD:
            return random.choices(
                ["took_action", "acknowledged", "ignored"],
                weights=[0.05, 0.15, 0.80]
            )[0]

        # Trust-based response probabilities
        if self.guardian_trust > 0.7:
            return random.choices(
                ["took_action", "acknowledged", "ignored"],
                weights=[0.50, 0.35, 0.15]
            )[0]
        elif self.guardian_trust > 0.4:
            return random.choices(
                ["took_action", "acknowledged", "ignored"],
                weights=[0.25, 0.40, 0.35]
            )[0]
        else:
            return random.choices(
                ["took_action", "acknowledged", "ignored"],
                weights=[0.10, 0.20, 0.70]
            )[0]