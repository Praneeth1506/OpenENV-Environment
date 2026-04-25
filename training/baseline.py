# training/baseline.py
# Run this FIRST before any training.
# Generates baseline JSON files that become the before curve on plots.
# Person B must run this before train_grpo.py

import sys
import os
import json
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server.environment import SafeSignalEnvironment, ACTIONS


def run_baseline(n_episodes=200, save_dir="../results"):
    """
    Runs random and silent agents for n_episodes each.
    Saves results to JSON files for plotting.

    These files are the before curve judges see.
    Run this BEFORE training. Never overwrite after training starts.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Running baseline evaluation — {n_episodes} episodes each\n")
    print("=" * 60)

    # ── Random Agent ───────────────────────────────────────────────
    print("POLICY: Random Agent")
    random_results = []

    for ep in range(n_episodes):
        env = SafeSignalEnvironment()
        result = env.reset()
        state = result.observation
        done = False
        ep_reward = 0
        trust_traj = []
        hidden_traj = []
        actions_taken = []

        while not done:
            action = random.choice(ACTIONS)
            step = env.step(action)
            ep_reward += step.reward
            trust_traj.append(step.info["guardian_trust"])
            hidden_traj.append(step.info["hidden_state"])
            actions_taken.append(action)
            state = step.observation
            done = step.done

        random_results.append({
            "episode": ep,
            "total_reward": round(ep_reward, 3),
            "final_hidden_state": step.info["hidden_state"],
            "final_guardian_trust": round(
                step.info["guardian_trust"], 3
            ),
            "trust_trajectory": trust_traj,
            "hidden_trajectory": hidden_traj,
            "total_interventions": sum(
                1 for a in actions_taken
                if a != "OBSERVE_QUIETLY"
            ),
            "ended_safe": step.info["hidden_state"] == "SAFE",
        })

        if ep % 50 == 0:
            avg = sum(
                r["total_reward"] for r in random_results
            ) / len(random_results)
            print(f"  Episode {ep:3d} | Avg reward: {avg:+.2f}")

    random_avg = sum(
        r["total_reward"] for r in random_results
    ) / len(random_results)
    random_safe_pct = sum(
        1 for r in random_results if r["ended_safe"]
    ) / len(random_results) * 100

    print(f"\n  Random avg reward:  {random_avg:+.2f}")
    print(f"  % ended safe:       {random_safe_pct:.1f}%")

    with open(f"{save_dir}/baseline_rewards.json", "w") as f:
        json.dump({
            "policy": "random",
            "n_episodes": n_episodes,
            "avg_reward": round(random_avg, 3),
            "pct_ended_safe": round(random_safe_pct, 1),
            "episodes": random_results,
        }, f, indent=2)
    print(f"  Saved: {save_dir}/baseline_rewards.json")

    # ── Always Silent Agent ────────────────────────────────────────
    print("\nPOLICY: Always Silent")
    silent_results = []

    for ep in range(n_episodes):
        env = SafeSignalEnvironment()
        result = env.reset()
        done = False
        ep_reward = 0
        trust_traj = []
        hidden_traj = []

        while not done:
            step = env.step("OBSERVE_QUIETLY")
            ep_reward += step.reward
            trust_traj.append(step.info["guardian_trust"])
            hidden_traj.append(step.info["hidden_state"])
            done = step.done

        silent_results.append({
            "episode": ep,
            "total_reward": round(ep_reward, 3),
            "final_hidden_state": step.info["hidden_state"],
            "final_guardian_trust": round(
                step.info["guardian_trust"], 3
            ),
            "trust_trajectory": trust_traj,
            "hidden_trajectory": hidden_traj,
            "total_interventions": 0,
            "ended_safe": step.info["hidden_state"] == "SAFE",
        })

        if ep % 50 == 0:
            avg = sum(
                r["total_reward"] for r in silent_results
            ) / len(silent_results)
            print(f"  Episode {ep:3d} | Avg reward: {avg:+.2f}")

    silent_avg = sum(
        r["total_reward"] for r in silent_results
    ) / len(silent_results)
    silent_safe_pct = sum(
        1 for r in silent_results if r["ended_safe"]
    ) / len(silent_results) * 100

    print(f"\n  Silent avg reward:  {silent_avg:+.2f}")
    print(f"  % ended safe:       {silent_safe_pct:.1f}%")

    with open(f"{save_dir}/silent_rewards.json", "w") as f:
        json.dump({
            "policy": "silent",
            "n_episodes": n_episodes,
            "avg_reward": round(silent_avg, 3),
            "pct_ended_safe": round(silent_safe_pct, 1),
            "episodes": silent_results,
        }, f, indent=2)
    print(f"  Saved: {save_dir}/silent_rewards.json")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print(f"  Random avg reward:   {random_avg:+.2f}")
    print(f"  Silent avg reward:   {silent_avg:+.2f}")
    print(f"  Trained must beat:   {silent_avg + 3.0:+.2f}")
    print(f"  Ideal ceiling:       +17.46 (from test_env.py)")
    print("=" * 60)
    print("\nBaseline complete. Share these files with Person C.")
    print("Now run: python train_grpo.py")

    return random_avg, silent_avg


if __name__ == "__main__":
    run_baseline(n_episodes=200)
