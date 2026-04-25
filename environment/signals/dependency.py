# signals/dependency.py
# Signal Cluster 5 — Emotional Dependency Formation
# Predator positions as the only person who understands the child.
# Creates measurable emotional correlation patterns.

import random

class DependencySignals:

    def __init__(self, hidden_state_numeric, archetype):
        self.risk = hidden_state_numeric
        self.archetype = archetype

    def compute(self):
        return {
            # Contact appears specifically during child's low moments
            # Rescue pattern is a documented grooming technique
            "rescue_pattern_score": self._rescue_pattern(),

            # Child's emotional tone tracks this contact's activity
            # 0.0 = no correlation (healthy)
            # 1.0 = child mood entirely dependent on contact (grooming)
            "emotional_dependency_score": self._emotional_dependency(),

            # Child confides to this contact but not existing friends
            "exclusive_confiding_shift": self._exclusive_confiding(),

            # Contact consistently provides positive emotional tone
            # Deliberately building reward association
            "contact_sentiment_provision_rate": self._sentiment_provision(),
        }

    def _rescue_pattern(self):
        base = self.risk * 0.22
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _emotional_dependency(self):
        base = self.risk * 0.28
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _exclusive_confiding(self):
        base = self.risk * 0.20
        noise = random.gauss(0, 0.04)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _sentiment_provision(self):
        # High even in safe state — predator is always positive
        base = 0.6 + (self.risk * 0.1)
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)