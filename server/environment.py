# server/environment.py
# Object-based API wrapper around SafeSignalEnv.
# Training scripts import SafeSignalEnvironment and ACTIONS from here.
#
# Adds environment/ to sys.path so the bare imports inside rubrics.py,
# simulated_child.py, and reward.py resolve correctly — this is required
# because those files use absolute imports (e.g. "from reward import ...").

import sys
import os

_ENV_DIR = os.path.join(os.path.dirname(__file__), '..', 'environment')
_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _ENV_DIR)
sys.path.insert(0, _ROOT_DIR)

from environment.safesignal_env import SafeSignalEnv, ACTIONS  # noqa: E402
from rubrics import SafeSignalRubricSystem                      # noqa: E402

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

    def get_episode_summary(self):
        return self._env.get_episode_summary()
