# signals/reciprocity.py
# Signal Cluster 1 — Reciprocity Imbalance
# Grooming creates asymmetric conversation patterns detectable
# without reading message content.

import random

class ReciprocitySignals:
    """
    Measures conversation balance between child and contact.
    Healthy friendships are symmetric.
    Grooming is asymmetric — predator always initiates, pursues, dominates.
    """

    def __init__(self, hidden_state_numeric, archetype):
        self.risk = hidden_state_numeric
        self.archetype = archetype

    def compute(self):
        return {
            # Who starts conversations
            # 0.0 = child always initiates (safe)
            # 1.0 = contact always initiates (grooming signal)
            "initiation_ratio": self._initiation_ratio(),

            # Contact messages are much longer than child messages
            # Predators invest more language to build rapport
            "message_length_ratio": self._message_length_ratio(),

            # Contact re-initiates after child goes silent
            # Normal friends accept silence. Predators pursue.
            "pursuit_score": self._pursuit_score(),

            # Contact responds much faster than child
            # Instant response = high investment = grooming signal
            "response_time_delta": self._response_time_delta(),
        }

    def _initiation_ratio(self):
        # Safe: balanced initiation ~0.5
        # AT_RISK: contact initiates ~0.85
        base = 0.5 + (self.risk * 0.15)
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _message_length_ratio(self):
        # Safe: similar length messages ~1.0
        # AT_RISK: contact sends 3x longer messages ~3.0
        base = 1.0 + (self.risk * 0.7)
        noise = random.gauss(0, 0.1)
        return round(max(0.5, base + noise), 3)

    def _pursuit_score(self):
        # Safe: contact accepts silence ~0.1
        # IN_DANGER: contact always pursues ~0.9
        base = 0.1 + (self.risk * 0.25)
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _response_time_delta(self):
        # How much faster contact responds vs child
        # Safe: similar response times ~0.0
        # AT_RISK: contact responds near-instantly, child slowly
        base = self.risk * 0.3
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)