# signals/secrecy.py
# Signal Cluster 4 — Secrecy Signal Cluster
# Grooming creates secrecy behaviors observable in metadata.

import random

class SecrecySignals:

    def __init__(self, hidden_state_numeric, archetype):
        self.risk = hidden_state_numeric
        self.archetype = archetype

    def compute(self):
        return {
            # Child responds faster to this contact at night
            # vs during day when family present
            "response_time_variance_by_hour": self._response_variance(),

            # Conversation gaps correlate with family presence
            "conversation_gap_pattern": self._gap_pattern(),

            # Change in how child handles notifications
            "notification_behavior_change": self._notification_change(),

            # Switching apps when family nearby
            "session_interruption_correlation": self._interruption(),
        }

    def _response_variance(self):
        base = self.risk * 0.25
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _gap_pattern(self):
        base = self.risk * 0.22
        noise = random.gauss(0, 0.04)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _notification_change(self):
        base = self.risk * 0.18
        noise = random.gauss(0, 0.04)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _interruption(self):
        base = self.risk * 0.20
        noise = random.gauss(0, 0.04)
        return round(min(1.0, max(0.0, base + noise)), 3)