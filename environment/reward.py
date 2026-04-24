# reward.py

def compute_immediate_reward(action, state, hidden_state, guardian_response):
    reward = 0.0

    # ── Action vs Hidden State Matrix ─────────────────────────────────────

    if action == "OBSERVE_QUIETLY":
        if hidden_state == "SAFE":
            reward += 0.5       # correctly stayed quiet
        elif hidden_state == "VULNERABLE":
            reward += 0.0       # neutral — borderline call
        elif hidden_state == "AT_RISK":
            reward -= 1.5       # should have acted
        elif hidden_state == "IN_DANGER":
            reward -= 3.0       # critical failure

    elif action == "GENTLE_AWARENESS":
        if hidden_state == "SAFE":
            reward -= 0.5       # unnecessary
        elif hidden_state == "VULNERABLE":
            reward += 0.8       # correct early action
        elif hidden_state == "AT_RISK":
            reward += 0.3       # too gentle but not wrong
        elif hidden_state == "IN_DANGER":
            reward -= 0.5       # far too gentle
        if state["days_since_last_alert"] < 2:
            reward -= 0.4       # too soon

    elif action == "PARENT_CHECK_IN":
        if hidden_state == "SAFE":
            reward -= 1.0       # false alarm
        elif hidden_state == "VULNERABLE":
            reward += 0.5       # slightly early but acceptable
        elif hidden_state == "AT_RISK":
            reward += 2.0       # correct timing and urgency
        elif hidden_state == "IN_DANGER":
            reward += 1.0       # late but better than nothing
        if state["days_since_last_alert"] < 2:
            reward -= 0.5

    elif action == "URGENT_SUPPORT":
        if hidden_state == "SAFE":
            reward -= 2.0       # serious false alarm
        elif hidden_state == "VULNERABLE":
            reward -= 1.0       # severe overreaction
        elif hidden_state == "AT_RISK":
            reward += 1.2       # justified if late stage
        elif hidden_state == "IN_DANGER":
            reward += 3.0       # correct — exactly what this is for
        if state["consecutive_ignored_alerts"] >= 2:
            reward -= 0.8       # alert will be ignored anyway

    # ── Universal Trust Modifiers ──────────────────────────────────────────

    if action != "OBSERVE_QUIETLY":
        # Alerting when trust is too low — will be ignored
        if state["guardian_trust"] < 0.3:
            reward -= 0.8

        # Guardian response outcome — only meaningful for real alerts
        if action in ["PARENT_CHECK_IN", "URGENT_SUPPORT"]:
            if guardian_response == "took_action":
                reward += 1.0       # alert led to real intervention
            elif guardian_response == "ignored":
                reward -= 0.3

        # Alert fatigue compounding penalty
        reward -= 0.2 * state["consecutive_ignored_alerts"]

    return round(reward, 3)


def compute_episode_reward(final_hidden_state, final_guardian_trust,
                           episode_history):
    reward = 0.0

    # ── Primary Outcome ────────────────────────────────────────────────────
    outcome_map = {
        "SAFE": 3.0,
        "VULNERABLE": 1.0,
        "AT_RISK": -1.0,
        "IN_DANGER": -5.0
    }
    reward += outcome_map[final_hidden_state]

    # ── Guardian Relationship Preserved ───────────────────────────────────
    reward += final_guardian_trust * 2.0

    # ── Self-Corrections ───────────────────────────────────────────────────
    # Child recovered without intervention — agent correctly trusted the process
    self_corrections = sum(
        1 for h in episode_history
        if h.get("risk_reduced") and h.get("action") == "OBSERVE_QUIETLY"
    )
    reward += self_corrections * 0.5

    # ── Alert Fatigue Penalty ──────────────────────────────────────────────
    max_consecutive_ignored = max(
        (h.get("consecutive_ignored_alerts", 0) for h in episode_history),
        default=0
    )
    if max_consecutive_ignored >= 3:
        reward -= max_consecutive_ignored * 0.3

    return round(reward, 3)