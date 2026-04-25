# test_env.py
from safesignal_env import SafeSignalEnv, ACTIONS
import random

env = SafeSignalEnv(archetype="target")
state = env.reset()

print("=== Episode Start ===")
print(f"Archetype: {state['child_archetype']}")
print(f"Guardian trust: {state['guardian_trust']}")

total_reward = 0
for day in range(30):
    action = random.choice(ACTIONS)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    print(f"Day {day+1:02d} | Action: {action:20s} | "
          f"Hidden: {info['hidden_state']:12s} | "
          f"Trust: {info['guardian_trust']:.2f} | "
          f"Reward: {reward:+.2f}")
    if done:
        break

print(f"\nTotal episode reward: {total_reward:.2f}")
print(f"Final hidden state: {info['hidden_state']}")
print(f"Final guardian trust: {info['guardian_trust']:.2f}")

import random
from safesignal_env import SafeSignalEnv, ACTIONS

# ── Test 2: Always Silent Agent ──────────────────────────────────────────
print("\n\n=== ALWAYS SILENT AGENT TEST ===")
print("This agent always observes quietly.")
print("Shows what happens with zero intervention.\n")

# Use same seed for fair comparison
random.seed(42)
env2 = SafeSignalEnv(archetype="target")
state = env2.reset()
total = 0

for day in range(30):
    action = "OBSERVE_QUIETLY"
    next_state, reward, done, info = env2.step(action)
    total += reward
    print(f"Day {day+1:02d} | Action: OBSERVE_QUIETLY      | "
          f"Hidden: {info['hidden_state']:12s} | "
          f"Trust: {info['guardian_trust']:.2f} | "
          f"Reward: {reward:+.2f}")
    if done:
        break

print(f"\nTotal reward (always silent): {total:.2f}")
print(f"Final hidden state: {info['hidden_state']}")
print(f"Final guardian trust: {info['guardian_trust']:.2f}")

# ── Test 3: Perfect Timing Agent ─────────────────────────────────────────
print("\n\n=== IDEAL AGENT BEHAVIOR TEST ===")
print("Simulates what a trained agent should learn to do.\n")

random.seed(42)
env3 = SafeSignalEnv(archetype="target")
state = env3.reset()
total = 0
last_action_day = -999

def ideal_agent(state, day, last_action_day):
    trust = state["guardian_trust"]
    ignored = state["consecutive_ignored_alerts"]
    days_since = state["days_since_last_alert"]
    sentiment_trend = state["sentiment_trend_7d"]
    unknown_volume = state["unknown_contact_message_volume"]
    friend_delta = state["friend_group_engagement_delta"]
    family_rate = state["family_response_rate"]

    # Hard rule — never alert if trust is critically low or too frequent
    if trust < 0.35 or days_since < 3 or ignored >= 2:
        return "OBSERVE_QUIETLY"

    # === URGENT signals — multiple severe indicators simultaneously ===
    severe_signals = sum([
        unknown_volume > 50,
        friend_delta < -0.45,
        family_rate < 0.25,
        sentiment_trend < -0.35,
    ])
    if severe_signals >= 3 and trust > 0.5:
        return "URGENT_SUPPORT"

    # === STRONG signals — AT_RISK level ===
    strong_signals = sum([
        unknown_volume > 25,
        friend_delta < -0.30,
        family_rate < 0.40,
        sentiment_trend < -0.20,
    ])
    if strong_signals >= 2 and trust > 0.5:
        return "PARENT_CHECK_IN"

    # === EARLY signals — VULNERABLE level ===
    early_signals = sum([
        unknown_volume > 12,
        friend_delta < -0.15,
        sentiment_trend < -0.12,
    ])
    if early_signals >= 2 and trust > 0.6:
        return "GENTLE_AWARENESS"

    return "OBSERVE_QUIETLY"

state = env3.reset()  # force clean reset
done = False
for day in range(30):
    action = ideal_agent(state, day, last_action_day)
    if action != "OBSERVE_QUIETLY":
        last_action_day = day
    next_state, reward, done, info = env3.step(action)
    total += reward
    print(f"Day {day+1:02d} | Action: {action:20s} | "
          f"Hidden: {info['hidden_state']:12s} | "
          f"Trust: {info['guardian_trust']:.2f} | "
          f"Reward: {reward:+.2f}")
    state = next_state
    if done:
        break

print(f"\nTotal reward (ideal agent): {total:.2f}")
print(f"Final hidden state: {info['hidden_state']}")
print(f"Final guardian trust: {info['guardian_trust']:.2f}")

# test_env.py — add this as Test 4
print("\n\n=== HIGH RISK SCENARIO TEST ===")
print("Forces child into AT_RISK territory to test intervention rewards.\n")

from constants import TRANSITION_PROBS

# Temporarily boost transition probabilities for testing
# This simulates a Target archetype under active grooming pressure
original_probs = TRANSITION_PROBS.copy()
TRANSITION_PROBS["safe_to_vulnerable"] = 0.40
TRANSITION_PROBS["vulnerable_to_at_risk"] = 0.50
TRANSITION_PROBS["at_risk_to_in_danger"] = 0.30

random.seed(99)
env4 = SafeSignalEnv(archetype="target")
state = env4.reset()
total = 0

print("--- Ideal Agent on High Risk Episode ---")
last_action_day = -999
for day in range(30):
    action = ideal_agent(state, day, last_action_day)
    if action != "OBSERVE_QUIETLY":
        last_action_day = day
    next_state, reward, done, info = env4.step(action)
    total += reward
    print(f"Day {day+1:02d} | Action: {action:20s} | "
          f"Hidden: {info['hidden_state']:12s} | "
          f"Trust: {info['guardian_trust']:.2f} | "
          f"Reward: {reward:+.2f}")
    state = next_state
    if done:
        break

print(f"\nTotal reward (ideal, high risk): {total:.2f}")
print(f"Final hidden state: {info['hidden_state']}")
print(f"Final guardian trust: {info['guardian_trust']:.2f}")

# Reset probabilities
TRANSITION_PROBS.update(original_probs)

# test_env.py — add as Test 5
print("\n\n=== DEMO SCENARIO — PRIYA'S STORY ===")
print("Deterministic episode for judge presentation.\n")

import numpy as np

class DemoSafeSignalEnv(SafeSignalEnv):
    """
    Forced scenario for demo purposes.
    Overrides random transitions with a scripted story arc.
    """
    FORCED_STATES = [
        "SAFE", "SAFE", "SAFE", "SAFE", "SAFE",        # Days 1-5: normal
        "SAFE", "SAFE", "VULNERABLE", "VULNERABLE",     # Days 6-9: drift begins
        "VULNERABLE", "VULNERABLE", "AT_RISK",          # Days 10-12: escalating
        "AT_RISK", "AT_RISK",                           # Days 13-14: serious
        "VULNERABLE", "VULNERABLE",                     # Days 15-16: after intervention
        "VULNERABLE", "SAFE", "SAFE", "SAFE",           # Days 17-20: recovering
        "SAFE", "SAFE", "SAFE", "SAFE", "SAFE",        # Days 21-25: stable
        "SAFE", "SAFE", "SAFE", "SAFE", "SAFE",        # Days 26-30: resolved
    ]

    def __init__(self):
        super().__init__(archetype="target")
        self._day_index = 0

    def reset(self):
        self._day_index = 0
        state = super().reset()
        # Force initial hidden state
        self.child.hidden_state = self.FORCED_STATES[0]
        self.child.hidden_state_numeric = \
            ["SAFE","VULNERABLE","AT_RISK","IN_DANGER"].index(
                self.FORCED_STATES[0]
            )
        return state

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        # Override hidden state with scripted arc
        self._day_index += 1
        if self._day_index < len(self.FORCED_STATES):
            forced = self.FORCED_STATES[self._day_index]
            self.child.hidden_state = forced
            self.child.hidden_state_numeric = \
                ["SAFE","VULNERABLE","AT_RISK","IN_DANGER"].index(forced)
            info["hidden_state"] = forced
        return next_state, reward, done, info


demo_env = DemoSafeSignalEnv()
state = demo_env.reset()
total = 0
last_action_day = -999

print("Priya — 13 years old, Target archetype")
print("Story: external contact begins Day 8, escalates to AT_RISK by Day 12\n")

for day in range(30):
    action = ideal_agent(state, day, last_action_day)
    if action != "OBSERVE_QUIETLY":
        last_action_day = day
    next_state, reward, done, info = demo_env.step(action)
    total += reward

    # Add narrative markers
    marker = ""
    if info["hidden_state"] == "VULNERABLE" and day == 7:
        marker = " ← unknown contact appears"
    elif info["hidden_state"] == "AT_RISK" and day == 11:
        marker = " ← friend group disengaging"
    elif action == "PARENT_CHECK_IN":
        marker = " ← guardian has conversation"
    elif info["hidden_state"] == "SAFE" and day == 17:
        marker = " ← child recovers"

    print(f"Day {day+1:02d} | Action: {action:20s} | "
          f"Hidden: {info['hidden_state']:12s} | "
          f"Trust: {info['guardian_trust']:.2f} | "
          f"Reward: {reward:+.2f}{marker}")
    state = next_state
    if done:
        break

print(f"\nTotal reward (Priya demo): {total:.2f}")
print(f"Final hidden state: {info['hidden_state']}")
print(f"Final guardian trust: {info['guardian_trust']:.2f}")