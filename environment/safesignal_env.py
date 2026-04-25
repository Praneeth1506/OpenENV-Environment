# safesignal_env.py
import random
import sys
import os

try:
    import openenv
    BASE_CLASS = openenv.env.Env
except ImportError:
    BASE_CLASS = object

from simulated_child import SimulatedChild
from reward import compute_immediate_reward, compute_episode_reward
from constants import EPISODE_LENGTH

ACTIONS = [
    "OBSERVE_QUIETLY",
    "GENTLE_AWARENESS",
    "PARENT_CHECK_IN",
    "URGENT_SUPPORT"
]

class SafeSignalEnv(BASE_CLASS):

    def __init__(self, archetype=None):
        if BASE_CLASS != object:
            super().__init__()
        self.archetype = archetype
        self.child = None
        self.day = 0
        self.episode_history = []

    def reset(self):
        """Start a fresh 30-day episode."""
        self.child = SimulatedChild(archetype=self.archetype)
        self.day = 0
        self.episode_history = []
        return self.child.get_observable_state()

    def step(self, action):
        """
        Takes agent action. Simulates one day.
        Returns (next_state, reward, done, info).
        """
        assert action in ACTIONS, f"Invalid action: {action}"

        # Get current state snapshot before advancing
        current_state = self.child.get_observable_state()
        current_hidden = self.child.hidden_state

        # Simulate guardian response to this action
        guardian_response = self.child.simulate_guardian_response(action)

        # Compute immediate reward
        immediate_reward = compute_immediate_reward(
            action=action,
            state=current_state,
            hidden_state=current_hidden,
            guardian_response=guardian_response
        )

        # Store history entry before advancing
        prev_hidden_numeric = self.child.hidden_state_numeric

        # Advance the child's day — updates hidden state and guardian trust
        self.child.advance_day(action, guardian_response)

        # Detect if risk reduced (for self-correction bonus)
        risk_reduced = self.child.hidden_state_numeric < prev_hidden_numeric

        # Log episode history
        self.episode_history.append({
            "day": self.day,
            "action": action,
            "hidden_state": current_hidden,
            "guardian_response": guardian_response,
            "guardian_trust": current_state["guardian_trust"],
            "consecutive_ignored_alerts": current_state["consecutive_ignored_alerts"],
            "risk_reduced": risk_reduced,
            "immediate_reward": immediate_reward,
        })

        self.day += 1
        done = self.day >= EPISODE_LENGTH

        # Add episode-level reward at end
        total_reward = immediate_reward
        if done:
            episode_bonus = compute_episode_reward(
                final_hidden_state=self.child.hidden_state,
                final_guardian_trust=self.child.guardian_trust,
                episode_history=self.episode_history
            )
            total_reward += episode_bonus

        next_state = self.child.get_observable_state()

        info = {
            "hidden_state": self.child.hidden_state,
            "guardian_trust": self.child.guardian_trust,
            "day": self.day,
            "guardian_response": guardian_response
        }

        return next_state, total_reward, done, info

    def get_episode_summary(self):
        """Returns episode history for demo visualization."""
        return self.episode_history