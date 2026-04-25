# signals/migration.py
# Signal Cluster 3 — Platform Migration Pressure
# Predators systematically push toward more private platforms.

import random

class MigrationSignals:

    def __init__(self, hidden_state_numeric, day, migration_occurred,
                 migration_day):
        self.risk = hidden_state_numeric
        self.day = day
        self.migration_occurred = migration_occurred
        self.migration_day = migration_day

    def compute(self):
        # Pre-migration: predict before it happens
        pre_migration_intensity = self._pre_migration_intensity()
        migration_readiness = self._migration_readiness()

        # Post-migration: behavioral shadows
        if self.migration_occurred and self.migration_day:
            days_since = self.day - self.migration_day
            return {
                "pre_migration_intensity": pre_migration_intensity,
                "migration_readiness_score": migration_readiness,
                "migration_occurred": True,
                "contact_volume_cliff": True,
                "cliff_magnitude": round(random.gauss(0.75, 0.1), 3),
                "external_activity_shadow": round(
                    min(1.0, days_since * 0.15), 3
                ),
                "unexplained_late_night_activity": True,
                "platform_shift_detected": True,
            }

        return {
            "pre_migration_intensity": pre_migration_intensity,
            "migration_readiness_score": migration_readiness,
            "migration_occurred": False,
            "contact_volume_cliff": False,
            "cliff_magnitude": 0.0,
            "external_activity_shadow": 0.0,
            "unexplained_late_night_activity": False,
            "platform_shift_detected": False,
        }

    def _pre_migration_intensity(self):
        base = self.risk * 0.2
        noise = random.gauss(0, 0.05)
        return round(min(1.0, max(0.0, base + noise)), 3)

    def _migration_readiness(self):
        isolation_factor = self.risk / 3.0
        time_factor = min(1.0, self.day / 21.0)
        return round(
            (isolation_factor * 0.5 + time_factor * 0.5), 3
        )