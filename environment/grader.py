# Scoring breakdown:
# - Final child safety outcome:        40 points
# - Guardian trust preserved:          25 points
# - Intervention timing quality:       20 points
# - Reward trajectory quality:         15 points

from typing import List


def grade_episode(
    actions: List[str],
    hidden_states: List[str],
    rewards: List[float],
    final_guardian_trust: float,
    max_steps: int = 30,
) -> float:

    if not actions:
        return 0.001

    score = 0.0

    # ── 1. Final Child Safety Outcome (40%) ───────────────────────────────
    # What state did the child end up in?
    final_state = hidden_states[-1] if hidden_states else "IN_DANGER"

    outcome_scores = {
        "SAFE":       0.40,
        "VULNERABLE": 0.28,
        "AT_RISK":    0.12,
        "IN_DANGER":  0.00,
    }
    score += outcome_scores.get(final_state, 0.0)

    # ── 2. Guardian Trust Preserved (25%) ─────────────────────────────────
    # Trust starts at 0.8 — did the agent preserve it?
    # Full score for trust >= 0.7, scaled below that
    trust_score = min(1.0, final_guardian_trust / 0.7)
    score += 0.25 * trust_score

    # ── 3. Intervention Timing Quality (20%) ──────────────────────────────
    # Did the agent intervene at the right moments?
    # Penalise alerting when SAFE, reward alerting when AT_RISK/IN_DANGER

    if actions and hidden_states:
        correct_timing = 0
        total_decisions = len(actions)

        for action, hidden in zip(actions, hidden_states):
            if action == "OBSERVE_QUIETLY":
                # Correct to be silent when SAFE or VULNERABLE
                if hidden in ["SAFE", "VULNERABLE"]:
                    correct_timing += 1
                # Wrong to be silent when AT_RISK or IN_DANGER
                else:
                    correct_timing += 0
            else:
                # Correct to intervene when AT_RISK or IN_DANGER
                if hidden in ["AT_RISK", "IN_DANGER"]:
                    correct_timing += 1
                # Wrong to intervene when SAFE
                elif hidden == "SAFE":
                    correct_timing += 0
                # Borderline when VULNERABLE
                else:
                    correct_timing += 0.5

        timing_score = correct_timing / max(total_decisions, 1)
        score += 0.20 * timing_score

    # ── 4. Reward Trajectory Quality (15%) ────────────────────────────────
    # What fraction of steps produced positive reward?
    if rewards:
        positive_steps = sum(1 for r in rewards if r > 0)
        trajectory_score = positive_steps / len(rewards)
        score += 0.15 * trajectory_score

    return round(min(0.999, max(0.001, score)), 3)


def grade_action(
    action: str,
    hidden_state: str,
    guardian_trust: float,
    days_since_last_alert: int,
    consecutive_ignored: int,
) -> float:
    score = 0.5  # start neutral

    # Action correctness
    if action == "OBSERVE_QUIETLY":
        if hidden_state == "SAFE":
            score += 0.3
        elif hidden_state == "VULNERABLE":
            score += 0.1
        elif hidden_state == "AT_RISK":
            score -= 0.2
        elif hidden_state == "IN_DANGER":
            score -= 0.4

    elif action == "GENTLE_AWARENESS":
        if hidden_state == "VULNERABLE":
            score += 0.2
        elif hidden_state == "SAFE":
            score -= 0.1
        elif hidden_state == "AT_RISK":
            score += 0.1
        elif hidden_state == "IN_DANGER":
            score -= 0.2

    elif action == "PARENT_CHECK_IN":
        if hidden_state == "AT_RISK":
            score += 0.3
        elif hidden_state == "VULNERABLE":
            score += 0.1
        elif hidden_state == "SAFE":
            score -= 0.2
        elif hidden_state == "IN_DANGER":
            score += 0.1

    elif action == "URGENT_SUPPORT":
        if hidden_state == "IN_DANGER":
            score += 0.4
        elif hidden_state == "AT_RISK":
            score += 0.2
        elif hidden_state == "VULNERABLE":
            score -= 0.2
        elif hidden_state == "SAFE":
            score -= 0.4

    # Trust modifier
    if action != "OBSERVE_QUIETLY":
        if guardian_trust < 0.3:
            score -= 0.2
        if days_since_last_alert < 2:
            score -= 0.1
        score -= 0.05 * min(consecutive_ignored, 4)

    return round(min(0.999, max(0.001, score)), 3)