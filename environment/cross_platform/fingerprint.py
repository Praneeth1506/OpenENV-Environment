# cross_platform/fingerprint.py
# Behavioral fingerprinting for cross-platform contact matching.
# Matches same contact across platforms without identity data.

class ContactFingerprint:
    """
    Identifies the same contact across different platforms
    using behavioral patterns, not account identity.

    Analogous to fraud detection behavioral fingerprinting —
    an established and legally sound technique.
    """

    def compute_fingerprint(self, contact_behavior):
        return {
            "active_hours_distribution": contact_behavior.get(
                "hour_histogram", []
            ),
            "response_time_mean": contact_behavior.get(
                "avg_response_seconds", 0
            ),
            "response_time_variance": contact_behavior.get(
                "response_std", 0
            ),
            "daily_message_volume_mean": contact_behavior.get(
                "daily_volume", 0
            ),
            "initiation_ratio": contact_behavior.get("who_starts", 0.5),
            "message_length_mean": contact_behavior.get("avg_length", 0),
            "weekend_vs_weekday_ratio": contact_behavior.get(
                "weekend_rate", 1.0
            ),
            "pursuit_score": contact_behavior.get(
                "reinitiates_after_silence", 0
            ),
            "intensity_trend": contact_behavior.get("volume_trend_14d", 0),
        }

    def similarity_score(self, fp1, fp2):
        """
        Computes similarity between two behavioral fingerprints.
        Returns (is_same_contact, confidence_score).
        Does not use any identity data.
        """
        scores = []

        # Response time similarity
        rt1 = fp1.get("response_time_mean", 0)
        rt2 = fp2.get("response_time_mean", 0)
        if max(rt1, rt2) > 0:
            rt_similarity = 1 - abs(rt1 - rt2) / max(rt1, rt2, 1)
            scores.append(rt_similarity)

        # Volume similarity
        v1 = fp1.get("daily_message_volume_mean", 0)
        v2 = fp2.get("daily_message_volume_mean", 0)
        if max(v1, v2) > 0:
            vol_similarity = min(v1, v2) / max(v1, v2)
            scores.append(vol_similarity)

        # Initiation pattern similarity
        i1 = fp1.get("initiation_ratio", 0.5)
        i2 = fp2.get("initiation_ratio", 0.5)
        init_similarity = 1 - abs(i1 - i2)
        scores.append(init_similarity)

        # Pursuit score similarity
        p1 = fp1.get("pursuit_score", 0)
        p2 = fp2.get("pursuit_score", 0)
        pursuit_similarity = 1 - abs(p1 - p2)
        scores.append(pursuit_similarity)

        if not scores:
            return False, 0.0

        confidence = sum(scores) / len(scores)
        threshold = 0.82
        return confidence > threshold, round(confidence, 3)