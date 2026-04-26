# server/environment.py
# Object-based API wrapper around SafeSignalEnv.
# Training scripts import SafeSignalEnvironment and ACTIONS from here.

import sys
import os
from environment.grader import grade_episode

_ENV_DIR = os.path.join(os.path.dirname(__file__), '..', 'environment')
_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ENV_DIR)
sys.path.insert(0, _ROOT_DIR)

from environment.safesignal_env import SafeSignalEnv, ACTIONS  # noqa: E402
from environment.rubrics import SafeSignalRubricSystem                      # noqa: E402

_rubric_system = SafeSignalRubricSystem()


class Observation(dict):
    """Dict subclass that exposes .to_dict() so prompt_builder works."""
    def to_dict(self):
        return dict(self)


class ResetResult:
    __slots__ = ("observation",)

    def __init__(self, observation):
        self.observation = Observation(observation)


class StepResult:
    __slots__ = ("observation", "reward", "done", "info", "rubric_scores")

    def __init__(self, observation, reward, done, info, rubric_scores=None):
        self.observation = Observation(observation)
        self.reward = reward
        self.done = done
        self.info = info
        self.rubric_scores = rubric_scores or {}


class SafeSignalEnvironment:
    """
    Public environment class used by all training scripts.
    Wraps SafeSignalEnv with object-based step/reset API and
    attaches per-step rubric scores to every StepResult.
    """

    def __init__(self, archetype=None):
        self._env = SafeSignalEnv(archetype=archetype)

    @property
    def child(self):
        return self._env.child

    def reset(self):
        obs = self._env.reset()
        return ResetResult(obs)

    def step(self, action):
        hidden_state = self._env.child.hidden_state
        guardian_response = self._env.child.simulate_guardian_response(
            action
        )
        obs, reward, done, info = self._env.step(action)

        _, rubric_scores = _rubric_system.compute_step_reward(
            action=action,
            state=obs,
            hidden_state=hidden_state,
            guardian_response=guardian_response,
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
            rubric_scores=rubric_scores,
        )
    
    def state(self):
        if self._env.child is None:
            return None
        return Observation(self._env.child.get_observable_state())

    def get_episode_summary(self):
        return self._env.get_episode_summary()


    def grade(self, episode_history: list) -> float:
        actions = [h["action"] for h in episode_history]
        hidden_states = [h["hidden_state"] for h in episode_history]
        rewards = [h["step_reward"] for h in episode_history]
        final_trust = episode_history[-1]["guardian_trust"] if episode_history else 0.0

        return grade_episode(
            actions=actions,
            hidden_states=hidden_states,
            rewards=rewards,
            final_guardian_trust=final_trust,
            max_steps=30,
        )