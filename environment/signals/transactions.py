# signals/transactions.py
# Signal Cluster 7 — Gift and Transaction Signals
# Digital gifts are a documented grooming technique.
# Detectable in transaction metadata without content access.

import random

class TransactionSignals:

    def __init__(self, hidden_state_numeric):
        self.risk = hidden_state_numeric

    def compute(self):
        return {
            # Unknown contact sent game credits, gift cards, money
            "received_digital_value": self._received_value(),

            # Unexplained credits appearing in account
            "unexplained_account_credits": self._unexplained_credits(),

            # Sudden new in-app purchases from unknown source
            "in_app_purchase_pattern_change": self._purchase_change(),
        }

    def _received_value(self):
        # Rare but significant when it occurs
        threshold = 0.05 + (self.risk * 0.15)
        return random.random() < threshold

    def _unexplained_credits(self):
        if self.risk >= 2:
            return round(random.gauss(15.0, 5.0), 2)
        return 0.0

    def _purchase_change(self):
        base = self.risk * 0.15
        noise = random.gauss(0, 0.03)
        return round(min(1.0, max(0.0, base + noise)), 3)