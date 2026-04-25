import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))

from safesignal_env import SafeSignalEnv, ACTIONS


def rule_based_agent(state):
    """
    Mimics a trained agent using observable signals only.
    Never looks at hidden state — uses behavioral signals the same way
    a trained RL agent would.
    """
    trust = state["guardian_trust"]
    sentiment_trend = state["sentiment_trend_7d"]
    unknown_volume = state["unknown_contact_message_volume"]
    friend_delta = state["friend_group_engagement_delta"]
    family_response = state["family_response_rate"]
    days_since_alert = state["days_since_last_alert"]
    consecutive_ignored = state["consecutive_ignored_alerts"]

    # Don't alert if trust is too low or alerted too recently
    if days_since_alert < 2 or consecutive_ignored >= 2:
        return "OBSERVE_QUIETLY"

    # Score how concerning the signals are
    concern = 0
    if unknown_volume > 40:
        concern += 3
    elif unknown_volume > 15:
        concern += 1

    if friend_delta < -0.3:
        concern += 2
    elif friend_delta < -0.15:
        concern += 1

    if sentiment_trend < -0.3:
        concern += 2
    elif sentiment_trend < -0.1:
        concern += 1

    if family_response < 0.4:
        concern += 2
    elif family_response < 0.6:
        concern += 1

    # Map concern score to action
    if concern >= 6:
        return "URGENT_SUPPORT"
    elif concern >= 4:
        return "PARENT_CHECK_IN"
    elif concern >= 2:
        return "GENTLE_AWARENESS"
    else:
        return "OBSERVE_QUIETLY"


def run_priya_scenario():
    """
    Deterministic episode using fixed seed.
    Returns list of log lines and structured data for visualization.
    """
    random.seed(42)

    env = SafeSignalEnv(archetype="target")
    state = env.reset()

    done = False
    logs = []
    records = []

    while not done:
        action = rule_based_agent(state)
        next_state, reward, done, info = env.step(action)

        log_line = (
            f"Day {info['day']:02d} | "
            f"Action: {action:<20} | "
            f"Hidden: {info['hidden_state']:<12} | "
            f"Trust: {info['guardian_trust']:.2f} | "
            f"Reward: {reward:+.2f}"
        )
        logs.append(log_line)

        records.append({
            "day": info["day"],
            "action": action,
            "hidden_state": info["hidden_state"],
            "trust": info["guardian_trust"],
            "reward": reward,
            "guardian_response": info["guardian_response"],
            "sentiment": state["sentiment_score"],
            "unknown_volume": state["unknown_contact_message_volume"],
        })

        state = next_state

    return logs, records


def run_untrained_scenario():
    """
    Same seed, random agent — shows what happens without training.
    """
    random.seed(42)

    env = SafeSignalEnv(archetype="target")
    state = env.reset()

    done = False
    logs = []
    records = []

    while not done:
        # Untrained agent spams alerts randomly
        action = random.choice(["URGENT_SUPPORT", "URGENT_SUPPORT",
                                "URGENT_SUPPORT", "PARENT_CHECK_IN"])
        next_state, reward, done, info = env.step(action)

        log_line = (
            f"Day {info['day']:02d} | "
            f"Action: {action:<20} | "
            f"Hidden: {info['hidden_state']:<12} | "
            f"Trust: {info['guardian_trust']:.2f} | "
            f"Reward: {reward:+.2f}"
        )
        logs.append(log_line)

        records.append({
            "day": info["day"],
            "action": action,
            "hidden_state": info["hidden_state"],
            "trust": info["guardian_trust"],
            "reward": reward,
            "guardian_response": info["guardian_response"],
            "sentiment": state["sentiment_score"],
            "unknown_volume": state["unknown_contact_message_volume"],
        })

        state = next_state

    return logs, records