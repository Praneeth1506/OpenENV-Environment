# training/evaluate.py
# Run after training to score the trained model.
# Generates the numbers that go in the README.

import sys
import os
import json
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server.environment import SafeSignalEnvironment, ACTIONS
from training.prompt_builder import state_to_prompt, parse_action
from unsloth import FastLanguageModel


def evaluate_model(model_path, n_episodes=50):
    """
    Scores trained model against environment.
    Returns metrics for README and plots.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    results = []
    trust_scores = []
    outcomes = []

    for ep in range(n_episodes):
        env = SafeSignalEnvironment()
        result = env.reset()
        state = result.observation
        done = False
        ep_reward = 0

        while not done:
            prompt = state_to_prompt(state)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=True,
                )

            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            action = parse_action(response)
            step = env.step(action)
            ep_reward += step.reward
            state = step.observation
            done = step.done

        results.append(ep_reward)
        trust_scores.append(step.info["guardian_trust"])
        outcomes.append(step.info["hidden_state"])

    avg_reward = sum(results) / len(results)
    avg_trust = sum(trust_scores) / len(trust_scores)
    pct_safe = sum(
        1 for o in outcomes if o == "SAFE"
    ) / len(outcomes) * 100

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print(f"  Episodes evaluated:  {n_episodes}")
    print(f"  Avg reward:          {avg_reward:+.2f}")
    print(f"  Avg final trust:     {avg_trust:.2f}")
    print(f"  % ended safe:        {pct_safe:.1f}%")
    print("=" * 50)
    print("\nComparison:")
    print(f"  Random baseline:     -45.76")
    print(f"  Silent baseline:     +15.85")
    print(f"  Trained model:       {avg_reward:+.2f}")
    print(f"  Improvement vs random: {avg_reward - (-45.76):+.2f}")

    return {
        "avg_reward": round(avg_reward, 3),
        "avg_trust": round(avg_trust, 3),
        "pct_safe": round(pct_safe, 1),
        "episode_rewards": results,
    }


if __name__ == "__main__":
    evaluate_model("./checkpoints/final")
