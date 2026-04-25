import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
from safesignal_env import SafeSignalEnv
import random, json, numpy as np
from pathlib import Path

VALID_ACTIONS = ["OBSERVE_QUIETLY", "GENTLE_AWARENESS",
                 "PARENT_CHECK_IN", "URGENT_SUPPORT"]
N_EPISODES = 100

def run_baseline():
    env = SafeSignalEnv()
    episode_rewards = []
    action_counts = {a: 0 for a in VALID_ACTIONS}
    final_hidden_states = []

    for n in range(1, N_EPISODES + 1):
        state = env.reset()
        episode_total = 0.0
        done = False
        
        while not done:
            action = random.choice(VALID_ACTIONS)
            state, reward, done, info = env.step(action)
            episode_total += float(reward)
            action_counts[action] += 1
            if done:
                final_hidden_states.append(info.get("hidden_state", ""))

        episode_rewards.append(round(episode_total, 4))
        
        if n % 10 == 0:
            print(f"Episode {n}/100 complete — reward: {episode_total:.2f}")

    mean_val = float(np.mean(episode_rewards))
    std_val = float(np.std(episode_rewards))
    min_val = float(np.min(episode_rewards))
    max_val = float(np.max(episode_rewards))

    print("--- Baseline Summary ---")
    print(f"Mean reward : {mean_val:.2f}")
    print(f"Std         : {std_val:.2f}")
    print(f"Min / Max   : {min_val:.2f} / {max_val:.2f}")
    print(f"Action distribution: {action_counts}")
    print("Saved → results/baseline_rewards.json")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "episode_rewards": episode_rewards,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "action_counts": action_counts,
        "final_hidden_states": final_hidden_states,
        "config": {
            "n_episodes": N_EPISODES,
            "agent": "random",
            "episode_length": 30
        }
    }

    with open(results_dir / "baseline_rewards.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    run_baseline()
