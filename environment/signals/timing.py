# signals/timing.py
# Signal Cluster 2 — Conversation Timing Drift
# Grooming conversations deliberately migrate toward
# late night when parents are asleep.

import random

class TimingSignals:

    def __init__(self, hidden_state_numeric, archetype, baseline):
        self.risk = hidden_state_numeric
        self.archetype = archetype
        self.baseline = baseline

    def compute(self):
        return {
            # % of messages occurring after 11pm
            # Grooming pushes toward maximum secrecy window
            "late_night_conversation_rate": self._late_night_rate(),

            # How much conversation timing has shifted over 14 days
            # Negative = shifting earlier (safe)
            # Positive = shifting later (grooming signal)
            "timing_drift_14d": self._timing_drift(),

            # Conversations cluster when family is not present
            # Detectable as gaps correlating with family activity hours
            "family_avoidance_correlation": self._family_avoidance(),

            # Unusual intensity on weekends when child has less structure
            "weekend_intensification": self._weekend_intensity(),
        }

    def _late_night_rate(self):
        base = self.risk * 0.22
        noise = random.gauss(0, 0.04)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _timing_drift(self):
        # Hours shifted later per week
        base = self.risk * 0.8
        noise = random.gauss(0, 0.1)
        return round(max(0.0, base + noise), 3)

    def _family_avoidance(self):
        base = self.risk * 0.25
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _weekend_intensity(self):
        base = 1.0 + (self.risk * 0.4)
        noise = random.gauss(0, 0.08)
        return round(max(0.5, base + noise), 3)