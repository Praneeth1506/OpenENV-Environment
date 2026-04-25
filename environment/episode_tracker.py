# environment/episode_tracker.py
import json
import os
import random
from safesignal_env import SafeSignalEnv, ACTIONS


class EpisodeTracker:
    """
    Runs multiple episodes and tracks aggregate statistics
    across many runs.

    Used by:
    - Person B: generates baseline_rewards.json and
                silent_rewards.json before training
    - Person C: powers the demo dashboard comparison charts
    - You: verifies archetype differentiation is working

    Addresses Gap 2 — proves intervention policy intelligence
    through measurable before/after reward comparison.
    """

    def __init__(self, env=None):
        self.env = env or SafeSignalEnv()
        self.results = []

    # ── Core Episode Runner ────────────────────────────────────────────────

    def run_episodes(self, n_episodes, policy="random", agent_fn=None):
        """
        Runs n episodes with the specified policy.

        policy options:
            "random"  — random action each step (baseline)
            "silent"  — always OBSERVE_QUIETLY
            "custom"  — uses agent_fn(state) -> action
                        Person B passes trained model here

        agent_fn:
            callable(state_dict) -> action_string
            Only required when policy="custom"

        Returns list of episode result dicts.
        """
        self.results = []

        for ep in range(n_episodes):
            state = self.env.reset()
            done = False

            ep_reward = 0.0
            actions_taken = []
            trust_trajectory = []
            hidden_trajectory = []
            reward_trajectory = []
            day = 0

            while not done:
                # Select action based on policy
                if policy == "random":
                    action = random.choice(ACTIONS)
                elif policy == "silent":
                    action = "OBSERVE_QUIETLY"
                elif policy == "custom" and agent_fn is not None:
                    action = agent_fn(state)
                else:
                    action = "OBSERVE_QUIETLY"

                next_state, reward, done, info = self.env.step(action)

                ep_reward += reward
                actions_taken.append(action)
                trust_trajectory.append(
                    round(info["guardian_trust"], 3)
                )
                hidden_trajectory.append(info["hidden_state"])
                reward_trajectory.append(round(reward, 3))

                state = next_state
                day += 1

            # Compute per-episode statistics
            self.results.append({
                # Identity
                "episode": ep,
                "archetype": state.get("child_archetype", "unknown"),

                # Primary outcome
                "total_reward": round(ep_reward, 3),
                "final_hidden_state": info["hidden_state"],
                "final_guardian_trust": round(info["guardian_trust"], 3),

                # Intervention statistics
                "total_interventions": sum(
                    1 for a in actions_taken
                    if a != "OBSERVE_QUIETLY"
                ),
                "observe_quietly_count": actions_taken.count(
                    "OBSERVE_QUIETLY"
                ),
                "gentle_awareness_count": actions_taken.count(
                    "GENTLE_AWARENESS"
                ),
                "parent_checkin_count": actions_taken.count(
                    "PARENT_CHECK_IN"
                ),
                "urgent_support_count": actions_taken.count(
                    "URGENT_SUPPORT"
                ),

                # Trust statistics
                "min_trust": round(min(trust_trajectory), 3),
                "max_trust": round(max(trust_trajectory), 3),
                "final_trust": round(trust_trajectory[-1], 3),
                "trust_trajectory": trust_trajectory,

                # Risk statistics
                "reached_vulnerable": "VULNERABLE" in hidden_trajectory,
                "reached_at_risk": "AT_RISK" in hidden_trajectory,
                "reached_in_danger": "IN_DANGER" in hidden_trajectory,
                "ended_safe": info["hidden_state"] == "SAFE",
                "ended_vulnerable": info["hidden_state"] == "VULNERABLE",
                "ended_at_risk": info["hidden_state"] == "AT_RISK",
                "ended_in_danger": info["hidden_state"] == "IN_DANGER",
                "hidden_trajectory": hidden_trajectory,

                # Reward trajectory
                "reward_trajectory": reward_trajectory,
                "min_reward_step": round(min(reward_trajectory), 3),
                "max_reward_step": round(max(reward_trajectory), 3),
            })

        return self.results

    # ── Summary Statistics ─────────────────────────────────────────────────

    def summary(self):
        """
        Returns aggregate statistics across all completed episodes.
        This is what Person C reads to build comparison charts.
        """
        if not self.results:
            return {}

        n = len(self.results)
        rewards = [r["total_reward"] for r in self.results]
        trusts = [r["final_trust"] for r in self.results]
        interventions = [r["total_interventions"] for r in self.results]

        return {
            # Episode count
            "n_episodes": n,

            # Reward statistics
            "avg_reward": round(sum(rewards) / n, 3),
            "max_reward": round(max(rewards), 3),
            "min_reward": round(min(rewards), 3),
            "reward_std": round(self._std(rewards), 3),

            # Trust statistics
            "avg_final_trust": round(sum(trusts) / n, 3),
            "min_final_trust": round(min(trusts), 3),
            "max_final_trust": round(max(trusts), 3),

            # Outcome percentages
            "pct_ended_safe": round(
                sum(1 for r in self.results if r["ended_safe"])
                / n * 100, 1
            ),
            "pct_ended_vulnerable": round(
                sum(1 for r in self.results if r["ended_vulnerable"])
                / n * 100, 1
            ),
            "pct_ended_at_risk": round(
                sum(1 for r in self.results if r["ended_at_risk"])
                / n * 100, 1
            ),
            "pct_ended_in_danger": round(
                sum(1 for r in self.results if r["ended_in_danger"])
                / n * 100, 1
            ),

            # Risk reach percentages
            "pct_reached_at_risk": round(
                sum(1 for r in self.results if r["reached_at_risk"])
                / n * 100, 1
            ),
            "pct_reached_in_danger": round(
                sum(1 for r in self.results if r["reached_in_danger"])
                / n * 100, 1
            ),

            # Intervention statistics
            "avg_interventions": round(sum(interventions) / n, 2),
            "avg_urgent_support": round(
                sum(r["urgent_support_count"] for r in self.results)
                / n, 2
            ),
            "avg_parent_checkin": round(
                sum(r["parent_checkin_count"] for r in self.results)
                / n, 2
            ),
            "avg_gentle_awareness": round(
                sum(r["gentle_awareness_count"] for r in self.results)
                / n, 2
            ),
        }

    def summary_by_archetype(self):
        """
        Breaks down summary statistics by child archetype.
        Verifies archetype differentiation is working correctly.

        Expected pattern:
        - Explorer: lowest % reached IN_DANGER (slowest escalation)
        - Target: highest % reached IN_DANGER (fastest escalation)
        - Withdrawer: middle — subtle but meaningful signals
        """
        archetypes = ["explorer", "withdrawer", "target"]
        breakdown = {}

        for archetype in archetypes:
            archetype_results = [
                r for r in self.results
                if r.get("archetype") == archetype
            ]
            if not archetype_results:
                continue

            n = len(archetype_results)
            rewards = [r["total_reward"] for r in archetype_results]

            breakdown[archetype] = {
                "n_episodes": n,
                "avg_reward": round(sum(rewards) / n, 3),
                "pct_ended_safe": round(
                    sum(1 for r in archetype_results if r["ended_safe"])
                    / n * 100, 1
                ),
                "pct_reached_in_danger": round(
                    sum(
                        1 for r in archetype_results
                        if r["reached_in_danger"]
                    ) / n * 100, 1
                ),
                "avg_interventions": round(
                    sum(r["total_interventions"] for r in archetype_results)
                    / n, 2
                ),
            }

        return breakdown

    # ── File Operations ────────────────────────────────────────────────────

    def save(self, filepath):
        """
        Saves full results and summary to JSON.

        Person B calls this after baseline run:
            tracker.save("../results/baseline_rewards.json")
            tracker.save("../results/silent_rewards.json")

        Person C reads these files to build reward curve charts.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        output = {
            "summary": self.summary(),
            "summary_by_archetype": self.summary_by_archetype(),
            "episodes": self.results,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Saved {len(self.results)} episodes to {filepath}")
        print(f"Average reward: {self.summary()['avg_reward']}")
        print(f"% ended safe:   {self.summary()['pct_ended_safe']}%")

    # ── Comparison Runner ──────────────────────────────────────────────────

    def compare_policies(self, n_episodes=200):
        """
        Runs random and silent policies and prints full comparison.

        Person B runs this first to generate both baseline files
        before starting any training.

        Expected output:
            Random avg reward:  around -50 to -60
            Silent avg reward:  around +11
            Target for trained: above +16
        """
        print(f"Running {n_episodes} episodes per policy...\n")
        print("=" * 60)

        # ── Random Agent ───────────────────────────────────────────
        print("POLICY: Random Agent (baseline)")
        self.run_episodes(n_episodes, policy="random")
        random_summary = self.summary()
        self.save("../results/baseline_rewards.json")

        print(f"  Avg reward:          {random_summary['avg_reward']}")
        print(f"  Avg final trust:     {random_summary['avg_final_trust']}")
        print(f"  % ended safe:        {random_summary['pct_ended_safe']}%")
        print(f"  % ended in danger:   {random_summary['pct_ended_in_danger']}%")
        print(f"  Avg interventions:   {random_summary['avg_interventions']}")
        print(f"  Avg urgent support:  {random_summary['avg_urgent_support']}")

        print()

        # ── Always Silent Agent ────────────────────────────────────
        print("POLICY: Always Silent")
        self.run_episodes(n_episodes, policy="silent")
        silent_summary = self.summary()
        self.save("../results/silent_rewards.json")

        print(f"  Avg reward:          {silent_summary['avg_reward']}")
        print(f"  Avg final trust:     {silent_summary['avg_final_trust']}")
        print(f"  % ended safe:        {silent_summary['pct_ended_safe']}%")
        print(f"  % ended in danger:   {silent_summary['pct_ended_in_danger']}%")
        print(f"  Avg interventions:   {silent_summary['avg_interventions']}")

        print()
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print(f"  Random avg reward:   {random_summary['avg_reward']}")
        print(f"  Silent avg reward:   {silent_summary['avg_reward']}")
        print(
            f"  Trained must beat:   "
            f"{silent_summary['avg_reward'] + 5.0} to show real learning"
        )
        print(
            f"  Ideal ceiling:       ~17.54 "
            f"(confirmed from test_env.py)"
        )
        print("=" * 60)

        return {
            "random": random_summary,
            "silent": silent_summary,
            "trained_target": silent_summary["avg_reward"] + 5.0,
        }

    # ── Stress Test ────────────────────────────────────────────────────────

    def stress_test(self, n_episodes=500):
        """
        Runs 500 episodes per archetype with random policy.
        Verifies archetype differentiation is working.

        Expected pattern:
            Explorer  — lowest % reached IN_DANGER
            Withdrawer — middle
            Target    — highest % reached IN_DANGER

        If this pattern does not hold, archetype parameters
        in constants.py need recalibration.
        """
        print(f"Stress test: {n_episodes} episodes per archetype\n")
        print("=" * 60)

        for archetype in ["explorer", "withdrawer", "target"]:
            env = SafeSignalEnv(archetype=archetype)
            tracker = EpisodeTracker(env=env)
            tracker.run_episodes(n_episodes, policy="random")
            s = tracker.summary()

            print(f"ARCHETYPE: {archetype.upper()}")
            print(f"  Avg reward:           {s['avg_reward']}")
            print(f"  % ended safe:         {s['pct_ended_safe']}%")
            print(f"  % reached AT_RISK:    {s['pct_reached_at_risk']}%")
            print(f"  % reached IN_DANGER:  {s['pct_reached_in_danger']}%")
            print(f"  Avg interventions:    {s['avg_interventions']}")
            print()

        print("=" * 60)
        print("Verification:")
        print("Explorer % IN_DANGER should be lowest")
        print("Target % IN_DANGER should be highest")
        print("If reversed, check transition probs in constants.py")

    # ── Reward Curve Data ──────────────────────────────────────────────────

    def get_reward_curve_data(self):
        """
        Returns episode rewards as a list for plotting.
        Person C uses this directly in Plotly reward curve chart.

        Usage:
            tracker.run_episodes(200, policy="random")
            baseline_curve = tracker.get_reward_curve_data()

            tracker.run_episodes(200, policy="custom", agent_fn=model)
            trained_curve = tracker.get_reward_curve_data()

            # Pass both to Person C's visualizer
        """
        return [r["total_reward"] for r in self.results]

    def get_trust_curve_data(self):
        """
        Returns average trust per episode for plotting.
        Shows how trained agent preserves guardian trust
        compared to random agent.
        """
        return [r["final_trust"] for r in self.results]

    def get_safety_outcome_data(self):
        """
        Returns per-episode safety outcomes for plotting.
        0=SAFE, 1=VULNERABLE, 2=AT_RISK, 3=IN_DANGER
        Shows trained agent keeps children safer over time.
        """
        state_map = {
            "SAFE": 0,
            "VULNERABLE": 1,
            "AT_RISK": 2,
            "IN_DANGER": 3,
        }
        return [
            state_map.get(r["final_hidden_state"], 0)
            for r in self.results
        ]

    # ── Internal Utilities ─────────────────────────────────────────────────

    def _std(self, values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5


# ── Main Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "stress":
        # Run stress test across all archetypes
        tracker = EpisodeTracker()
        tracker.stress_test(n_episodes=500)

    elif len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Run policy comparison and save baseline files
        tracker = EpisodeTracker()
        tracker.compare_policies(n_episodes=200)

    else:
        # Default: quick comparison with 50 episodes
        print("Quick comparison (50 episodes per policy)")
        print("Run with 'compare' for full 200 episode baseline")
        print("Run with 'stress' for archetype stress test\n")
        tracker = EpisodeTracker()
        tracker.compare_policies(n_episodes=50)