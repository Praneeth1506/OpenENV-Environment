# cross_platform/migration_detector.py
# Three-layer cross-platform migration detection.

from cross_platform.fingerprint import ContactFingerprint

class MigrationDetector:
    """
    Three-layer detection architecture.

    Layer 1: Within Meta ecosystem (Instagram/WhatsApp/Messenger)
    Layer 2: Pre-migration prediction before it happens
    Layer 3: External platform behavioral shadow detection
    """

    def __init__(self):
        self.fingerprinter = ContactFingerprint()
        self.known_contacts = {}
        self.migration_events = []

    def check_layer_1(self, instagram_contacts, whatsapp_contacts):
        """
        Within Meta ecosystem — volume shift detection.
        Contact fingerprint appears on new platform while
        volume drops on original platform.
        """
        results = []
        for ig_contact_id, ig_behavior in instagram_contacts.items():
            ig_fp = self.fingerprinter.compute_fingerprint(ig_behavior)
            for wa_contact_id, wa_behavior in whatsapp_contacts.items():
                wa_fp = self.fingerprinter.compute_fingerprint(wa_behavior)
                is_same, confidence = self.fingerprinter.similarity_score(
                    ig_fp, wa_fp
                )
                if is_same:
                    volume_shift = (
                        wa_behavior.get("daily_volume", 0) /
                        max(ig_behavior.get("daily_volume", 1), 1)
                    )
                    if volume_shift > 1.5:
                        results.append({
                            "type": "layer_1_migration",
                            "confidence": confidence,
                            "volume_shift": volume_shift,
                            "from_platform": "instagram",
                            "to_platform": "whatsapp",
                        })
        return results

    def check_layer_2(self, migration_readiness_score, threshold=0.75):
        """
        Pre-migration prediction.
        Flags migration risk before the attempt happens.
        """
        if migration_readiness_score > threshold:
            return {
                "type": "layer_2_prediction",
                "migration_predicted": True,
                "readiness_score": migration_readiness_score,
                "confidence": migration_readiness_score,
                "action": "intervene_before_migration",
            }
        return {"migration_predicted": False}

    def check_layer_3(self, behavioral_shadow_score, threshold=0.5):
        """
        External platform shadow detection.
        Behavior changing without visible on-platform cause.
        """
        if behavioral_shadow_score > threshold:
            return {
                "type": "layer_3_shadow",
                "external_migration_likely": True,
                "shadow_score": behavioral_shadow_score,
                "signals": [
                    "sentiment_declining_without_contact",
                    "unexplained_late_night_activity",
                    "contact_volume_cliff",
                ],
            }
        return {"external_migration_likely": False}