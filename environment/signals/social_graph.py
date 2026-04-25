# signals/social_graph.py
# Signal Cluster 6 — Social Graph Compression
# Isolation is gradual and measurable.
# One of the earliest and most consistent signals in research.

import random

class SocialGraphSignals:

    def __init__(self, hidden_state_numeric, archetype, baseline):
        self.risk = hidden_state_numeric
        self.archetype = archetype
        self.baseline = baseline

    def compute(self):
        return {
            # Total active contacts is shrinking
            "active_contact_count_trend": self._contact_count_trend(),

            # % of all communication going to one unknown contact
            "single_contact_concentration": self._concentration(),

            # Rate at which normal friendships are declining
            "existing_friendship_decay_rate": self._friendship_decay(),

            # Overall social circle diversity shrinking
            "contact_diversity_score": self._diversity(),
        }

    def _contact_count_trend(self):
        # Negative = shrinking social circle
        base = -(self.risk * 0.15)
        noise = random.gauss(0, 0.03)
        return round(base + noise, 3)

    def _concentration(self):
        base = self.risk * 0.25
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _friendship_decay(self):
        base = self.risk * 0.12
        noise = random.gauss(0, 0.03)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _diversity(self):
        # Decreases as risk increases
        base = self.baseline["friend_engagement_baseline"] - (self.risk * 0.2)
        noise = random.gauss(0, 0.04)
        return round(min(1.0, max(0.0, base + noise)), 3)