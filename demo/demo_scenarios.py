import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from safesignal_env import SafeSignalEnv, ACTIONS
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "env_demo_scenarios",
    os.path.join(os.path.dirname(__file__), '..', 'environment', 'demo_scenarios.py')
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DemoSafeSignalEnv = _mod.DemoSafeSignalEnv


# ── Rule-based agent (stand-in until trained model loaded) ────────────────

def rule_based_agent(state):
    """
    Uses observable signals only — never hidden state.
    Mimics what a trained agent should learn to do.
    """
    trust = state["guardian_trust"]
    ignored = state["consecutive_ignored_alerts"]
    days_since = state["days_since_last_alert"]
    vol = state["unknown_contact_message_volume"]
    delta = state["friend_group_engagement_delta"]
    family = state["family_response_rate"]
    sentiment = state["sentiment_trend_7d"]

    if days_since < 2 or ignored >= 2 or trust < 0.3:
        return "OBSERVE_QUIETLY"

    concern = 0
    if vol > 40:        concern += 3
    elif vol > 15:      concern += 1
    if delta < -0.3:    concern += 2
    elif delta < -0.15: concern += 1
    if sentiment < -0.3:  concern += 2
    elif sentiment < -0.1: concern += 1
    if family < 0.4:    concern += 2
    elif family < 0.6:  concern += 1

    if concern >= 6:    return "URGENT_SUPPORT"
    elif concern >= 4:  return "PARENT_CHECK_IN"
    elif concern >= 2:  return "GENTLE_AWARENESS"
    else:               return "OBSERVE_QUIETLY"


def random_agent(state):
    """Untrained — spams urgent alerts, destroys trust."""
    return random.choice([
        "URGENT_SUPPORT", "URGENT_SUPPORT",
        "URGENT_SUPPORT", "PARENT_CHECK_IN"
    ])


# ── Core episode runner ───────────────────────────────────────────────────

def run_episode(env, agent_fn, seed=42):
    random.seed(seed)
    state = env.reset()
    logs = []
    records = []
    done = False

    while not done:
        action = agent_fn(state)
        next_state, reward, done, info = env.step(action)

        logs.append(
            f"Day {info['day']:02d} | "
            f"Action: {action:<20} | "
            f"Hidden: {info['hidden_state']:<12} | "
            f"Trust: {info['guardian_trust']:.2f} | "
            f"Reward: {reward:+.2f}"
        )
        records.append({
            "day": info["day"],
            "action": action,
            "hidden_state": info["hidden_state"],
            "trust": info["guardian_trust"],
            "reward": reward,
            "guardian_response": info["guardian_response"],
            "sentiment": state.get("sentiment_score", 0),
            "unknown_volume": state.get("unknown_contact_message_volume", 0),
        })
        state = next_state

    return logs, records


# ── Public scenario functions ─────────────────────────────────────────────

def run_priya_scenario():
    """Trained agent on Priya's story."""
    env = DemoSafeSignalEnv()
    return run_episode(env, rule_based_agent, seed=42)


def run_untrained_scenario():
    """Random agent — shows alert fatigue collapse."""
    env = DemoSafeSignalEnv()
    return run_episode(env, random_agent, seed=42)


def run_live_episode(archetype="target"):
    """Random seed episode for interactive demo tab."""
    env = SafeSignalEnv(archetype=archetype)
    return run_episode(env, rule_based_agent, seed=random.randint(0, 9999))