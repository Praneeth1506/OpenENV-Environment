# cross_platform/shadow_signals.py
# Post-migration behavioral shadows.
# When conversation moves to external platform,
# the child's behavior on the original platform
# still reflects what is happening elsewhere.

import random

class ShadowSignals:
    """
    Detects behavioral shadows of off-platform conversations.
    The conversation is invisible but its effects are not.
    """

    def __init__(self, days_since_migration, risk_level):
        self.days_since = days_since_migration
        self.risk = risk_level

    def compute(self):
        shadow_intensity = min(1.0, self.days_since * 0.12)

        return {
            # Sentiment declining without any visible on-platform cause
            "sentiment_declining_without_contact": (
                shadow_intensity > 0.3
            ),

            # Active late at night but not messaging on platform
            "unexplained_late_night_activity": (
                shadow_intensity > 0.2
            ),

            # Family communication dropping with no on-platform explanation
            "family_response_unexplained_decline": (
                shadow_intensity > 0.4
            ),

            # Overall shadow intensity score
            "shadow_intensity": round(shadow_intensity, 3),

            # Confidence that external conversation is occurring
            "external_conversation_confidence": round(
                shadow_intensity * 0.85, 3
            ),
        }