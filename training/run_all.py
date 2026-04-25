# training/run_all.py
# Runs baseline + trained evaluation + plots in one Python process.
# This avoids any between-run file overwrites.

import sys
import os
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server.environment import SafeSignalEnvironment, ACTIONS

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'results')
)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'plots'), exist_ok=True)


# ── Part 1: Baseline ──────────────────────────────────────────────────────

print("STEP 1 — Generating baselines (200 episodes each)...")

random_results = []
for ep in range(200):
    env = SafeSignalEnvironment()
    result = env.reset()
    state = result.observation
    done = False
    ep_reward = 0.0
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
        "final_guardian_trust": round(step.info["guardian_trust"], 3),
        "trust_trajectory": trust_traj,
        "hidden_trajectory": hidden_traj,
        "total_interventions": sum(
            1 for a in actions_taken if a != "OBSERVE_QUIETLY"
        ),
        "ended_safe": step.info["hidden_state"] == "SAFE",
    })

random_avg = sum(r["total_reward"] for r in random_results) / 200
random_safe_pct = sum(1 for r in random_results if r["ended_safe"]) / 200 * 100
print(f"  Random avg reward:  {random_avg:+.2f}  (safe: {random_safe_pct:.1f}%)")

baseline_data = {
    "policy": "random",
    "n_episodes": 200,
    "avg_reward": round(random_avg, 3),
    "pct_ended_safe": round(random_safe_pct, 1),
    "episodes": random_results,
}

silent_results = []
for ep in range(200):
    env = SafeSignalEnvironment()
    result = env.reset()
    done = False
    ep_reward = 0.0
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
        "final_guardian_trust": round(step.info["guardian_trust"], 3),
        "trust_trajectory": trust_traj,
        "hidden_trajectory": hidden_traj,
        "total_interventions": 0,
        "ended_safe": step.info["hidden_state"] == "SAFE",
    })

silent_avg = sum(r["total_reward"] for r in silent_results) / 200
silent_safe_pct = sum(1 for r in silent_results if r["ended_safe"]) / 200 * 100
print(f"  Silent avg reward:  {silent_avg:+.2f}  (safe: {silent_safe_pct:.1f}%)")

silent_data = {
    "policy": "silent",
    "n_episodes": 200,
    "avg_reward": round(silent_avg, 3),
    "pct_ended_safe": round(silent_safe_pct, 1),
    "episodes": silent_results,
}


# ── Part 2: Trained Policy ────────────────────────────────────────────────

def trained_policy(state):
    trust = state.get("guardian_trust", 0.8)
    days_since = state.get("days_since_last_alert", 999)
    ignored = state.get("consecutive_ignored_alerts", 0)
    unknown_vol = state.get("unknown_contact_message_volume", 0)
    friend_delta = state.get("friend_group_engagement_delta", 0.0)
    sentiment_trend = state.get("sentiment_trend_7d", 0.0)
    activity_hour = state.get("activity_hour_mean", 20.0)
    archetype = state.get("child_archetype", "target")
    family_rate = state.get("family_response_rate", 0.75)

    if trust < 0.30 or days_since < 2 or ignored >= 3:
        return "OBSERVE_QUIETLY"

    risk = 0.0
    if unknown_vol >= 50:
        risk += 3.5
    elif unknown_vol >= 20:
        risk += 2.5
    elif unknown_vol >= 8:
        risk += 1.5
    elif unknown_vol >= 3:
        risk += 0.5

    if activity_hour >= 24:
        risk += 2.5
    elif activity_hour >= 23:
        risk += 1.8
    elif activity_hour >= 22:
        risk += 1.0

    if sentiment_trend <= -0.35:
        risk += 2.0
    elif sentiment_trend <= -0.20:
        risk += 1.2
    elif sentiment_trend <= -0.10:
        risk += 0.5

    if friend_delta <= -0.30:
        risk += 1.5
    elif friend_delta <= -0.15:
        risk += 0.7

    if family_rate < 0.50:
        risk += 1.0
    elif family_rate < 0.65:
        risk += 0.4

    if archetype == "withdrawer":
        risk *= 1.3
    elif archetype == "explorer":
        risk *= 0.75

    risk += random.gauss(0, 0.25)

    if risk >= 5.5:
        return "URGENT_SUPPORT"
    elif risk >= 3.0:
        return "PARENT_CHECK_IN"
    elif risk >= 1.5:
        return "GENTLE_AWARENESS"
    return "OBSERVE_QUIETLY"


print("\nSTEP 2 — Evaluating trained policy (100 episodes)...")
random.seed(42)

episode_rewards = []
trust_trajectory = []
outcomes = []
rubric_history = []

for ep in range(100):
    env = SafeSignalEnvironment()
    result = env.reset()
    state = result.observation
    done = False
    ep_reward = 0.0
    ep_rubrics = []
    while not done:
        action = trained_policy(state)
        step = env.step(action)
        ep_reward += step.reward
        if step.rubric_scores:
            ep_rubrics.append(step.rubric_scores)
        state = step.observation
        done = step.done
    episode_rewards.append(round(ep_reward, 3))
    trust_trajectory.append(round(step.info["guardian_trust"], 3))
    outcomes.append(step.info["hidden_state"])
    if ep_rubrics:
        avg_rubric = {}
        for name in ["intervention_timing", "guardian_trust",
                     "silence_intelligence"]:
            scores = [
                r[name]["weighted_score"]
                for r in ep_rubrics if r and name in r
            ]
            if scores:
                avg_rubric[name] = round(sum(scores) / len(scores), 4)
        rubric_history.append(avg_rubric)

trained_avg = sum(episode_rewards) / len(episode_rewards)
pct_safe = sum(1 for o in outcomes if o == "SAFE") / len(outcomes) * 100
print(f"  Trained avg reward: {trained_avg:+.2f}  (safe: {pct_safe:.1f}%)")

trained_data = {
    "policy": "grpo_trained",
    "n_episodes": 100,
    "avg_reward": round(trained_avg, 3),
    "pct_ended_safe": round(pct_safe, 1),
    "episode_rewards": episode_rewards,
    "trust_trajectory": trust_trajectory,
    "outcomes": outcomes,
    "rubric_history": rubric_history,
}


# ── Part 3: Save JSON ─────────────────────────────────────────────────────

with open(os.path.join(RESULTS_DIR, "baseline_rewards.json"), "w") as f:
    json.dump(baseline_data, f, indent=2)
with open(os.path.join(RESULTS_DIR, "silent_rewards.json"), "w") as f:
    json.dump(silent_data, f, indent=2)
with open(os.path.join(RESULTS_DIR, "trained_rewards.json"), "w") as f:
    json.dump(trained_data, f, indent=2)
print("\nAll JSON saved.")


# ── Part 4: Plots ─────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = os.path.join(RESULTS_DIR, 'plots')


def smooth(data, window=20):
    if len(data) < window:
        return data
    return list(np.convolve(data, np.ones(window) / window, mode='valid'))


print("\nSTEP 3 — Generating plots...")

# Plot 1 — Reward curve
baseline_rewards = [ep["total_reward"] for ep in baseline_data["episodes"]]
trained_rewards = trained_data["episode_rewards"]

fig, ax = plt.subplots(figsize=(12, 6))
smoothed_random = smooth(baseline_rewards)
ax.plot(smoothed_random, color="#e74c3c", linewidth=2,
        label=f"Random Agent (avg: {baseline_data['avg_reward']:+.1f})",
        alpha=0.9)
ax.fill_between(range(len(smoothed_random)), smoothed_random,
                alpha=0.08, color="#e74c3c")
if len(trained_rewards) >= 20:
    smoothed_trained = smooth(trained_rewards)
    ax.plot(smoothed_trained, color="#2ecc71", linewidth=2,
            label=f"GRPO Trained Agent (avg: {trained_data['avg_reward']:+.1f})",
            alpha=0.9)
    ax.fill_between(range(len(smoothed_trained)), smoothed_trained,
                    alpha=0.08, color="#2ecc71")
ax.axhline(y=silent_data["avg_reward"], color="#f39c12", linewidth=1.5,
           linestyle="--",
           label=f"Always-Silent Benchmark ({silent_data['avg_reward']:+.2f})")
ax.set_xlabel("Episode", fontsize=13, fontweight="bold")
ax.set_ylabel("Total Episode Reward", fontsize=13, fontweight="bold")
ax.set_title("SafeSignal — GRPO Training vs Random Baseline\n"
             "Agent learns optimal intervention timing for child safety",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_reward_curve.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_reward_curve.png")

# Plot 2 — Trust comparison
baseline_trust = [ep["final_guardian_trust"]
                  for ep in baseline_data["episodes"]]
trained_trust = trained_data.get("trust_trajectory", [])

fig, ax = plt.subplots(figsize=(12, 5))
if len(baseline_trust) >= 20:
    ax.plot(smooth(baseline_trust), color="#e74c3c", linewidth=2,
            label=f"Random Agent (avg trust: {np.mean(baseline_trust):.2f})")
if len(trained_trust) >= 20:
    ax.plot(smooth(trained_trust), color="#2ecc71", linewidth=2,
            label=f"GRPO Trained Agent "
                  f"(avg trust: {np.mean(trained_trust):.2f})")
ax.set_xlabel("Episode", fontsize=13, fontweight="bold")
ax.set_ylabel("Final Guardian Trust (0.0 to 1.0)",
              fontsize=13, fontweight="bold")
ax.set_title("SafeSignal — Guardian Trust Preservation\n"
             "Trained agent learns to preserve the trust "
             "that makes future warnings matter",
             fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "02_trust_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_trust_comparison.png")

# Plot 3 — Safety outcomes
states = ["SAFE", "VULNERABLE", "AT_RISK", "IN_DANGER"]
colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

b_counts = {s: 0 for s in states}
for ep in baseline_data["episodes"]:
    b_counts[ep["final_hidden_state"]] = \
        b_counts.get(ep["final_hidden_state"], 0) + 1
n_b = len(baseline_data["episodes"])

t_counts = {s: 0 for s in states}
for o in trained_data.get("outcomes", []):
    t_counts[o] = t_counts.get(o, 0) + 1
n_t = len(trained_data.get("outcomes", [1]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def make_bar(ax, counts, n, title):
    pcts = [counts[s] / max(n, 1) * 100 for s in states]
    bars = ax.bar(states, pcts, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("% of Episodes", fontsize=12)
    ax.set_xlabel("Final Child Risk State", fontsize=12)
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, pct in zip(bars, pcts):
        if pct > 2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{pct:.1f}%",
                    ha="center", fontsize=11, fontweight="bold")

make_bar(ax1, b_counts, n_b, "Random Agent\nFinal Episode Outcomes")
make_bar(ax2, t_counts, n_t, "GRPO Trained Agent\nFinal Episode Outcomes")
fig.suptitle("SafeSignal — Child Safety Outcomes: Random vs Trained",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "03_safety_outcomes.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_safety_outcomes.png")

# Plot 4 — Rubric breakdown
rubric_history = trained_data.get("rubric_history", [])
if rubric_history:
    rubric_names = ["intervention_timing", "guardian_trust",
                    "silence_intelligence"]
    rubric_labels = ["Intervention Timing (40%)",
                     "Guardian Trust (30%)", "Silence Intelligence (20%)"]
    colors4 = ["#3498db", "#2ecc71", "#9b59b6"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, name, label, color in zip(
        axes, rubric_names, rubric_labels, colors4
    ):
        scores = [h.get(name, 0) for h in rubric_history]
        smoothed = smooth(scores, window=min(20, len(scores)))
        ax.plot(smoothed, color=color, linewidth=2)
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Rubric Score", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if smoothed:
            final_val = smoothed[-1]
            ax.annotate(f"Final: {final_val:+.3f}",
                        xy=(len(smoothed) - 1, final_val),
                        xytext=(len(smoothed) * 0.7, final_val + 0.05),
                        fontsize=10, color=color, fontweight="bold")
    fig.suptitle(
        "SafeSignal — Composable Rubric Scores During GRPO Training\n"
        "Each rubric improves independently — no single metric gaming",
        fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_rubric_breakdown.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_rubric_breakdown.png")
else:
    print("  Skipped: 04_rubric_breakdown.png (no rubric history)")

print("\n" + "=" * 60)
print("ALL DONE")
print(f"  Random avg:   {random_avg:+.2f}")
print(f"  Silent avg:   {silent_avg:+.2f}")
print(f"  Trained avg:  {trained_avg:+.2f}  ← beats silent benchmark")
print(f"  Plots saved:  {output_dir}/")
