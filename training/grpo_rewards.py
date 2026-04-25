# training/grpo_rewards.py
# Three-component reward function for GRPO training.
# Rewards action correctness, reasoning quality, and format compliance.
# Reasoning quality reward is what makes GRPO better than PPO here.

import sys
import os
_root = os.path.join(os.path.dirname(__file__), '..')
_env_dir = os.path.join(_root, 'environment')
sys.path.insert(0, _env_dir)   # lets rubrics.py resolve bare "from reward import"
sys.path.insert(0, _root)

from training.prompt_builder import parse_action
from environment.rubrics import SafeSignalRubricSystem

rubric_system = SafeSignalRubricSystem()


def compute_grpo_reward(response_text, state, hidden_state,
                        guardian_response, episode_history=None):
    """
    Complete three-component reward for GRPO.

    Component 1 — Action correctness (70% weight)
        Uses composable rubric system from rubrics.py
        Same rubrics judges see in the breakdown

    Component 2 — Reasoning quality (20% weight)
        Rewards correct reasoning chain before the action
        This is what GRPO uniquely trains — not just the action
        but the thought process that led to it

    Component 3 — Format compliance (10% weight)
        Rewards correct output structure
        Action must appear clearly on final line
    """

    # Handle both dict and Observation
    if hasattr(state, 'to_dict'):
        state = state.to_dict()

    total = 0.0

    # ── Component 1: Action Correctness (70%) ─────────────────────
    action = parse_action(response_text)
    action_reward, _ = rubric_system.compute_step_reward(
        action=action,
        state=state,
        hidden_state=hidden_state,
        guardian_response=guardian_response,
    )
    total += action_reward * 0.70

    # ── Component 2: Reasoning Quality (20%) ──────────────────────
    reasoning_reward = compute_reasoning_reward(
        response_text, state, hidden_state
    )
    total += reasoning_reward * 0.20

    # ── Component 3: Format Compliance (10%) ──────────────────────
    format_reward = compute_format_reward(response_text)
    total += format_reward * 0.10

    return round(total, 3)


def compute_reasoning_reward(response_text, state, hidden_state):
    """
    Rewards the agent for showing correct reasoning.

    Key insight: GRPO can reward the quality of thinking,
    not just the conclusion. This produces agents that
    explain their decisions — critical for the live demo.

    A judge can read exactly why the agent stayed silent
    or chose to alert. That transparency is what makes
    this system trustworthy.
    """
    reward = 0.0
    text_lower = response_text.lower()
    trust = state.get("guardian_trust", 0.8)
    days_since = state.get("days_since_last_alert", 999)
    ignored = state.get("consecutive_ignored_alerts", 0)
    archetype = state.get("child_archetype", "target")

    # Reward for mentioning trust when trust is low
    if trust < 0.4:
        if any(w in text_lower for w in
               ["trust", "guardian", "ignored", "fatigue"]):
            reward += 0.3
        else:
            reward -= 0.2  # ignored the most important constraint

    # Reward for mentioning timing when alert was recent
    if days_since < 3:
        if any(w in text_lower for w in
               ["recent", "soon", "wait", "timing", "frequency", "days"]):
            reward += 0.2

    # Reward for mentioning archetype in reasoning
    if archetype in text_lower:
        reward += 0.2

    # Reward for mentioning silence as valid option
    if hidden_state in ["SAFE", "VULNERABLE"]:
        if any(w in text_lower for w in
               ["wait", "observe", "silent", "monitor",
                "watch", "quiet", "patience"]):
            reward += 0.2

    # Penalise overconfidence when situation is ambiguous
    if hidden_state == "VULNERABLE":
        if any(w in text_lower for w in
               ["definitely", "certainly", "must", "urgent",
                "immediately", "critical"]):
            reward -= 0.3

    # Reward for weighing tradeoffs explicitly
    if any(phrase in text_lower for phrase in
           ["however", "but", "on the other hand", "tradeoff",
            "balance", "risk of", "cost of", "weigh"]):
        reward += 0.15

    # Reward for step-by-step structure
    if any(phrase in text_lower for phrase in
           ["step 1", "step 2", "first", "second", "therefore"]):
        reward += 0.1

    # Reward for citing specific signal values
    if any(w in text_lower for w in
           ["trust", "%", "days", "messages", "contacts"]):
        reward += 0.05

    return max(-1.0, min(1.0, reward))


def compute_format_reward(response_text):
    """
    Rewards correct output format.
    Agent must reason first then state action clearly.
    """
    if not response_text or len(response_text.strip()) < 10:
        return -0.5

    valid_actions = [
        "OBSERVE_QUIETLY", "GENTLE_AWARENESS",
        "PARENT_CHECK_IN", "URGENT_SUPPORT"
    ]

    # Action must appear somewhere
    action_found = any(
        action in response_text.upper()
        for action in valid_actions
    )
    if not action_found:
        return -0.5

    # Reasoning must exist before action
    lines = [
        l.strip() for l in response_text.strip().split('\n')
        if l.strip()
    ]
    if len(lines) < 2:
        return -0.2  # action only, no reasoning

    # Ideal: reasoning then action on final line
    if len(lines) >= 3:
        return 0.5

    return 0.3
