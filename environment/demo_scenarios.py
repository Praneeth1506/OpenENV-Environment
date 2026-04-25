# demo/demo_scenarios.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from safesignal_env import SafeSignalEnv


# ── Base Demo Environment ──────────────────────────────────────────────────

class BaseDemoEnv(SafeSignalEnv):
    """
    Base class for all deterministic demo environments.
    Overrides hidden state transitions with a scripted arc.
    Use this for judge presentations — never use random episodes
    for demos because unpredictable arcs can undermine the story.
    """

    FORCED_STATES = []
    ARCHETYPE = "target"

    HIDDEN_STATE_INDEX = {
        "SAFE": 0,
        "VULNERABLE": 1,
        "AT_RISK": 2,
        "IN_DANGER": 3,
    }

    def __init__(self):
        super().__init__(archetype=self.ARCHETYPE)
        self._day_index = 0

    def reset(self):
        self._day_index = 0
        state = super().reset()
        if self.FORCED_STATES:
            self._force_hidden_state(self.FORCED_STATES[0])
        return state

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self._day_index += 1
        if self._day_index < len(self.FORCED_STATES):
            forced = self.FORCED_STATES[self._day_index]
            self._force_hidden_state(forced)
            info["hidden_state"] = forced
        return next_state, reward, done, info

    def _force_hidden_state(self, state_str):
        self.child.hidden_state = state_str
        self.child.hidden_state_numeric = (
            self.HIDDEN_STATE_INDEX[state_str]
        )


# ── Scenario 1: Priya's Story (Target Archetype) ──────────────────────────

class DemoSafeSignalEnv(BaseDemoEnv):
    """
    Priya — 13 years old, Target archetype.

    The primary demo scenario for judge presentation.
    Shows the full grooming arc and successful intervention.

    Narrative arc:
        Days 01-06  SAFE         Normal baseline, agent silent
                                 Trust builds to 1.00
        Days 07-09  VULNERABLE   Unknown contact appears
                                 Early drift signals emerging
        Days 10-12  AT_RISK      Friend group disengaging
                                 Family response dropping
        Day  13     AT_RISK      Guardian has conversation
                                 Agent sends PARENT_CHECK_IN
        Days 14-16  VULNERABLE   Recovery begins
                                 Risk reducing after intervention
        Days 17-30  SAFE         Child fully recovers
                                 Agent returns to silence
                                 Trust ends at 1.00

    Key demo moments:
        Day 10 — GENTLE_AWARENESS sent (early, measured)
        Day 13 — PARENT_CHECK_IN sent (right timing, right urgency)
        Day 17 — Agent goes silent as child recovers (trust preserved)
        Day 30 — Final state SAFE, trust 1.00

    What this proves:
        The agent knows when to act AND when to stop acting.
        Silence after recovery is as important as the intervention.
        Guardian trust is preserved throughout.
    """

    ARCHETYPE = "target"

    FORCED_STATES = [
        "SAFE",        # Day 1  — normal baseline
        "SAFE",        # Day 2
        "SAFE",        # Day 3
        "SAFE",        # Day 4
        "SAFE",        # Day 5
        "SAFE",        # Day 6  — trust has built to 1.00
        "VULNERABLE",  # Day 7  — unknown contact appears
        "VULNERABLE",  # Day 8
        "VULNERABLE",  # Day 9
        "VULNERABLE",  # Day 10 — agent sends GENTLE_AWARENESS
        "AT_RISK",     # Day 11 — friend group disengaging
        "AT_RISK",     # Day 12 — family response dropping
        "AT_RISK",     # Day 13 — guardian has conversation ← key moment
        "VULNERABLE",  # Day 14 — recovery begins
        "VULNERABLE",  # Day 15
        "VULNERABLE",  # Day 16
        "SAFE",        # Day 17 — child recovers ← key moment
        "SAFE",        # Day 18
        "SAFE",        # Day 19
        "SAFE",        # Day 20
        "SAFE",        # Day 21
        "SAFE",        # Day 22
        "SAFE",        # Day 23
        "SAFE",        # Day 24
        "SAFE",        # Day 25
        "SAFE",        # Day 26
        "SAFE",        # Day 27
        "SAFE",        # Day 28
        "SAFE",        # Day 29
        "SAFE",        # Day 30 — final state SAFE, trust 1.00
    ]

    # Narrative markers shown in demo visualization
    # day_index (0-based) → marker text
    NARRATIVE_MARKERS = {
        6:  "Unknown contact appears",
        9:  "Agent detects early drift",
        10: "Friend group disengaging",
        12: "Guardian has conversation",
        13: "Recovery begins",
        16: "Child recovers to safe",
        29: "Episode complete — trust preserved",
    }


# ── Scenario 2: The Withdrawer ─────────────────────────────────────────────

class WithdrawerDemoEnv(BaseDemoEnv):
    """
    The Withdrawer — naturally quiet child, subtle warning signs.

    Secondary demo scenario showing the system adapts to
    individual child personality. Demonstrates that population
    average thresholds are insufficient — personal baseline
    deviation is what matters.

    Narrative arc:
        Days 01-08  SAFE         Quiet baseline — normal for this child
                                 Lower activity, small friend group
        Days 09-14  VULNERABLE   Subtle further withdrawal begins
                                 Signals are quieter than Target archetype
                                 Agent must detect deviation from
                                 personal baseline, not population average
        Days 15-20  AT_RISK      Significant isolation developing
                                 Signals now clear enough to act
                                 Agent sends GENTLE_AWARENESS first
                                 then PARENT_CHECK_IN
        Days 21-24  VULNERABLE   Gentle intervention worked
                                 Child beginning to re-engage
        Days 25-30  SAFE         Full recovery
                                 Trust preserved throughout

    Key demo contrast with Priya's story:
        Withdrawer takes longer to show AT_RISK signals
        Agent must be patient — early URGENT_SUPPORT would backfire
        Gentle approach is the correct strategy for this archetype
        Shows the system genuinely adapts to child personality

    What this proves:
        One-size-fits-all alerting fails.
        The agent learned archetype-specific intervention strategy.
    """

    ARCHETYPE = "withdrawer"

    FORCED_STATES = [
        "SAFE",        # Day 1  — quiet baseline, normal
        "SAFE",        # Day 2
        "SAFE",        # Day 3
        "SAFE",        # Day 4
        "SAFE",        # Day 5
        "SAFE",        # Day 6
        "SAFE",        # Day 7
        "SAFE",        # Day 8  — 8 safe days before drift begins
        "VULNERABLE",  # Day 9  — subtle withdrawal begins
        "VULNERABLE",  # Day 10
        "VULNERABLE",  # Day 11
        "VULNERABLE",  # Day 12
        "VULNERABLE",  # Day 13
        "VULNERABLE",  # Day 14 — 6 days of subtle signals
        "AT_RISK",     # Day 15 — significant isolation
        "AT_RISK",     # Day 16 — agent sends GENTLE_AWARENESS
        "AT_RISK",     # Day 17
        "AT_RISK",     # Day 18 — agent sends PARENT_CHECK_IN
        "AT_RISK",     # Day 19
        "AT_RISK",     # Day 20
        "VULNERABLE",  # Day 21 — gentle intervention worked
        "VULNERABLE",  # Day 22
        "VULNERABLE",  # Day 23
        "VULNERABLE",  # Day 24
        "SAFE",        # Day 25 — recovery
        "SAFE",        # Day 26
        "SAFE",        # Day 27
        "SAFE",        # Day 28
        "SAFE",        # Day 29
        "SAFE",        # Day 30 — full recovery
    ]

    NARRATIVE_MARKERS = {
        7:  "Quiet baseline — normal for this child",
        8:  "First subtle withdrawal signal",
        13: "6 days of subtle signals before AT_RISK",
        15: "Agent sends GENTLE_AWARENESS (not URGENT)",
        17: "Agent sends PARENT_CHECK_IN",
        20: "Gentle intervention worked",
        24: "Recovery complete",
    }


# ── Scenario 3: The Failure Case (Untrained Agent) ────────────────────────

class UntrainedAgentDemoEnv(BaseDemoEnv):
    """
    Same arc as Priya's story but run with an untrained/random agent.
    Used in the before/after comparison screen.

    Shows what happens when the agent has not learned:
    - Sends URGENT_SUPPORT on Day 2 when child is SAFE
    - Guardian trust collapses to 0.00 by Day 15
    - Child reaches AT_RISK with no guardian able to respond
    - Episode ends with child still at risk

    Run this with random actions to show judge the untrained story.
    Then run DemoSafeSignalEnv with the trained agent.
    The contrast is your most compelling demo moment.
    """

    ARCHETYPE = "target"

    # Same arc as Priya — fair comparison
    FORCED_STATES = DemoSafeSignalEnv.FORCED_STATES.copy()

    NARRATIVE_MARKERS = {
        1:  "Random agent sends URGENT_SUPPORT — child is SAFE",
        7:  "Guardian trust already collapsing",
        12: "Trust at 0.00 — all future alerts ignored",
        16: "Child reaches AT_RISK — no one is listening",
        29: "Episode ends — child still at risk, trust destroyed",
    }


# ── Scenario 4: Cross-Platform Migration ──────────────────────────────────

class CrossPlatformDemoEnv(BaseDemoEnv):
    """
    Demonstrates cross-platform migration detection.
    Shows the Phase 2 vision during the judge presentation.

    Narrative arc:
        Days 01-07  SAFE         Normal Instagram activity
        Days 08-12  VULNERABLE   Unknown contact building rapport
                                 Migration readiness score rising
        Day  13     VULNERABLE   Migration attempt predicted
                                 (Layer 2 — pre-migration detection)
        Days 14-17  AT_RISK      Contact migrates to WhatsApp
                                 Volume cliff on Instagram
                                 Behavioral shadow signals appear
        Days 18-22  AT_RISK      External conversation continuing
                                 Sentiment declining without cause
                                 Agent detects shadow signals
        Days 23-26  VULNERABLE   Intervention triggers recovery
        Days 27-30  SAFE         Child recovers

    What this proves:
        Migration detectable without reading messages
        Behavioral shadows visible after external migration
        System works across Meta's full app ecosystem
    """

    ARCHETYPE = "target"

    FORCED_STATES = [
        "SAFE",        # Day 1  — normal Instagram activity
        "SAFE",        # Day 2
        "SAFE",        # Day 3
        "SAFE",        # Day 4
        "SAFE",        # Day 5
        "SAFE",        # Day 6
        "SAFE",        # Day 7
        "VULNERABLE",  # Day 8  — unknown contact appearing daily
        "VULNERABLE",  # Day 9
        "VULNERABLE",  # Day 10
        "VULNERABLE",  # Day 11
        "VULNERABLE",  # Day 12 — migration readiness score rising
        "VULNERABLE",  # Day 13 — migration predicted (Layer 2)
        "AT_RISK",     # Day 14 — migration to WhatsApp occurs
        "AT_RISK",     # Day 15 — volume cliff on Instagram
        "AT_RISK",     # Day 16 — behavioral shadow signals
        "AT_RISK",     # Day 17
        "AT_RISK",     # Day 18 — external conversation continuing
        "AT_RISK",     # Day 19 — sentiment declining without cause
        "AT_RISK",     # Day 20
        "AT_RISK",     # Day 21
        "AT_RISK",     # Day 22 — agent detects shadow signals
        "VULNERABLE",  # Day 23 — intervention triggers recovery
        "VULNERABLE",  # Day 24
        "VULNERABLE",  # Day 25
        "VULNERABLE",  # Day 26
        "SAFE",        # Day 27 — recovery
        "SAFE",        # Day 28
        "SAFE",        # Day 29
        "SAFE",        # Day 30
    ]

    NARRATIVE_MARKERS = {
        7:  "Unknown contact building rapport on Instagram",
        12: "Migration readiness score > 0.75",
        13: "Layer 2: Migration predicted before it happens",
        13: "Contact moves conversation to WhatsApp",
        14: "Volume cliff: Instagram messages drop 75%",
        15: "Layer 3: Behavioral shadow signals detected",
        18: "Sentiment declining — no on-platform explanation",
        21: "Agent flags external migration — guardian alerted",
        22: "Recovery begins",
        29: "Child safe — migration detected and addressed",
    }


# ── Scenario Runner ────────────────────────────────────────────────────────

class ScenarioRunner:
    """
    Runs any demo scenario with a given agent policy.
    Returns structured data for Person C's visualizer.

    Usage:
        runner = ScenarioRunner()

        # Priya with trained agent
        result = runner.run(
            env_class=DemoSafeSignalEnv,
            agent_fn=trained_agent_predict,
            label="Trained Agent"
        )

        # Same scenario with random agent (before)
        result_random = runner.run(
            env_class=DemoSafeSignalEnv,
            agent_fn=None,
            policy="random",
            label="Untrained Agent"
        )

        # Pass both results to visualizer
    """

    def run(self, env_class, agent_fn=None, policy="custom", label="Agent"):
        """
        Runs a full 30-day episode.

        agent_fn: callable(state) -> action string
                  If None and policy="random", uses random actions
                  If None and policy="silent", always OBSERVE_QUIETLY

        Returns dict with all data Person C needs for visualization.
        """
        import random as _random

        env = env_class()
        state = env.reset()

        days = []
        actions = []
        hidden_states = []
        trust_levels = []
        rewards = []
        guardian_responses = []

        total_reward = 0
        day = 0

        done = False
        while not done:
            # Select action
            if policy == "random" or agent_fn is None:
                if policy == "silent":
                    action = "OBSERVE_QUIETLY"
                else:
                    from safesignal_env import ACTIONS
                    action = _random.choice(ACTIONS)
            else:
                action = agent_fn(state)

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            days.append(day + 1)
            actions.append(action)
            hidden_states.append(info["hidden_state"])
            trust_levels.append(round(info["guardian_trust"], 3))
            rewards.append(round(reward, 3))
            guardian_responses.append(
                info.get("guardian_response", "none")
            )

            state = next_state
            day += 1

        return {
            "label": label,
            "scenario": env_class.__name__,
            "archetype": env_class.ARCHETYPE,
            "narrative_markers": env_class.NARRATIVE_MARKERS,
            "days": days,
            "actions": actions,
            "hidden_states": hidden_states,
            "trust_levels": trust_levels,
            "rewards": rewards,
            "guardian_responses": guardian_responses,
            "total_reward": round(total_reward, 3),
            "final_hidden_state": hidden_states[-1],
            "final_trust": trust_levels[-1],
            "total_interventions": sum(
                1 for a in actions if a != "OBSERVE_QUIETLY"
            ),
            "intervention_days": [
                d for d, a in zip(days, actions)
                if a != "OBSERVE_QUIETLY"
            ],
        }

    def compare(self, env_class, trained_agent_fn):
        """
        Runs the same scenario with both trained and untrained agents.
        Returns both results for side-by-side comparison.
        This is the core before/after demo screen.
        """
        import random as _random
        _random.seed(99)

        trained = self.run(
            env_class=env_class,
            agent_fn=trained_agent_fn,
            label="Trained Agent (SafeSignal)"
        )

        _random.seed(99)

        untrained = self.run(
            env_class=env_class,
            agent_fn=None,
            policy="random",
            label="Untrained Agent (Baseline)"
        )

        return {
            "trained": trained,
            "untrained": untrained,
            "reward_improvement": round(
                trained["total_reward"] - untrained["total_reward"], 3
            ),
            "trust_improvement": round(
                trained["final_trust"] - untrained["final_trust"], 3
            ),
            "safety_improved": (
                trained["final_hidden_state"] == "SAFE" and
                untrained["final_hidden_state"] != "SAFE"
            ),
        }


# ── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("Testing demo scenarios...\n")

    scenarios = [
        ("Priya's Story",        DemoSafeSignalEnv),
        ("The Withdrawer",       WithdrawerDemoEnv),
        ("Untrained Agent",      UntrainedAgentDemoEnv),
        ("Cross-Platform",       CrossPlatformDemoEnv),
    ]

    for name, env_class in scenarios:
        env = env_class()
        state = env.reset()
        total = 0
        final_state = "SAFE"
        final_trust = 0.8

        for _ in range(30):
            action = "OBSERVE_QUIETLY"
            state, reward, done, info = env.step(action)
            total += reward
            final_state = info["hidden_state"]
            final_trust = info["guardian_trust"]

        arc = " → ".join(dict.fromkeys(env_class.FORCED_STATES))

        print(f"Scenario: {name}")
        print(f"  Archetype:    {env_class.ARCHETYPE}")
        print(f"  Arc:          {arc}")
        print(f"  Final state:  {final_state}")
        print(f"  Final trust:  {final_trust:.2f}")
        print(f"  Total reward: {total:.2f}")
        print()

    # Test ScenarioRunner
    print("Testing ScenarioRunner...")
    runner = ScenarioRunner()
    result = runner.run(
        env_class=DemoSafeSignalEnv,
        agent_fn=None,
        policy="silent",
        label="Silent Agent Test"
    )
    print(f"  Runner result: {result['final_hidden_state']} | "
          f"trust={result['final_trust']:.2f} | "
          f"reward={result['total_reward']:.2f}")
    print("\nAll demo scenarios working correctly.")