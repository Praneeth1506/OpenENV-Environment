# training/train_grpo.py

import sys
import os
import json
import torch
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from safesignal_env import SafeSignalEnv, ACTIONS
from grpo_rewards import compute_grpo_reward
from prompt_builder import state_to_prompt, parse_action

# ── Step 1: Load Model with Unsloth ───────────────────────────────────────

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# ── Step 2: GRPO Configuration ────────────────────────────────────────────

from trl import GRPOConfig, GRPOTrainer

grpo_config = GRPOConfig(
    # Core GRPO parameters
    num_generations=8,          # G — number of responses per prompt
                                # More = better gradient estimate, more memory
                                # 8 is good for 1B model on Colab
    
    max_prompt_length=512,      # state prompt length
    max_completion_length=200,  # agent reasoning + action
    
    # Training parameters
    learning_rate=5e-6,         # lower than PPO — GRPO is more stable
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    
    # KL penalty — keeps model from drifting too far from base
    beta=0.01,                  # low beta = more freedom to explore
                                # high beta = stays closer to base model
    
    # Output
    output_dir="./grpo_checkpoints",
    logging_steps=10,
    save_steps=100,
    
    # Optimization
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
)

# ── Step 3: Build Training Dataset ───────────────────────────────────────
# GRPO needs a dataset of prompts to train on.
# We generate these by running the environment.

def generate_training_data(n_episodes=500):
    """
    Runs episodes to collect (prompt, hidden_state, state) tuples.
    GRPO will generate multiple responses per prompt and score them.
    """
    env = SafeSignalEnv()
    training_samples = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            prompt = state_to_prompt(state)
            
            # Store the true hidden state alongside the prompt
            # This is used for reward computation after GRPO generates responses
            training_samples.append({
                "prompt": prompt,
                "hidden_state": env.child.hidden_state,
                "state_snapshot": state.copy(),
                "episode": episode,
            })
            
            # Random action to advance the episode
            action = random.choice(ACTIONS)
            next_state, reward, done, info = env.step(action)
            state = next_state
    
    return training_samples

print("Generating training data...")
training_data = generate_training_data(n_episodes=500)
print(f"Generated {len(training_data)} training samples")

# Convert to HuggingFace dataset format
from datasets import Dataset

dataset = Dataset.from_list([
    {"prompt": sample["prompt"]} 
    for sample in training_data
])

# Store metadata separately for reward computation
training_metadata = {
    i: {
        "hidden_state": training_data[i]["hidden_state"],
        "state_snapshot": training_data[i]["state_snapshot"],
    }
    for i in range(len(training_data))
}

# ── Step 4: Reward Function Wrapper for GRPO ─────────────────────────────
# GRPO trainer expects a function: (prompts, completions) -> rewards

sample_index = [0]  # tracks which sample we are scoring

def grpo_reward_fn(prompts, completions, **kwargs):
    """
    Called by GRPOTrainer after generating completions.
    Returns list of reward scores, one per completion.
    """
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # Find matching metadata
        idx = sample_index[0] % len(training_metadata)
        meta = training_metadata[idx]
        
        # Parse the agent's action from completion
        response_text = completion[0]["content"] if isinstance(
            completion, list
        ) else completion
        
        action = parse_action(response_text)
        hidden_state = meta["hidden_state"]
        state = meta["state_snapshot"]
        
        # Simulate guardian response for reward computation
        env_temp = SafeSignalEnv()
        env_temp.child.hidden_state = hidden_state
        env_temp.child.guardian_trust = state["guardian_trust"]
        env_temp.child.consecutive_ignored_alerts = state[
            "consecutive_ignored_alerts"
        ]
        guardian_response = env_temp.child.simulate_guardian_response(action)
        
        # Compute full reward
        reward = compute_grpo_reward(
            response_text=response_text,
            state=state,
            hidden_state=hidden_state,
            guardian_response=guardian_response,
            episode_history=[],
        )
        
        rewards.append(float(reward))
    
    sample_index[0] += 1
    return rewards

# ── Step 5: Initialize and Run GRPO Trainer ───────────────────────────────

trainer = GRPOTrainer(
    model=model,
    reward_funcs=grpo_reward_fn,
    args=grpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("Starting GRPO training...")
print(f"Training on {len(dataset)} prompts")
print(f"Generating {grpo_config.num_generations} responses per prompt")
print(f"Effective batch size: {grpo_config.per_device_train_batch_size * grpo_config.gradient_accumulation_steps}")

# Run training
trainer.train()

# Save final model
trainer.save_model("./safesignal_grpo_final")
print("Training complete. Model saved.")

# ── Step 6: Evaluate Trained Model ───────────────────────────────────────

def evaluate_trained_model(model, tokenizer, n_episodes=50):
    """
    Runs trained model through episodes and computes average reward.
    Compare against baseline to show improvement.
    """
    env = SafeSignalEnv()
    episode_rewards = []
    
    FastLanguageModel.for_inference(model)
    
    for episode in range(n_episodes):
        state = env.reset()
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
                    temperature=0.1,   # low temperature = more consistent
                    do_sample=True,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            action = parse_action(response)
            
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state
        
        episode_rewards.append(ep_reward)
        
        if episode % 10 == 0:
            avg = sum(episode_rewards) / len(episode_rewards)
            print(f"Episode {episode} | Avg reward: {avg:.2f}")
    
    return {
        "avg_reward": sum(episode_rewards) / len(episode_rewards),
        "max_reward": max(episode_rewards),
        "min_reward": min(episode_rewards),
        "n_episodes": n_episodes,
    }

print("\nEvaluating trained model...")
results = evaluate_trained_model(model, tokenizer, n_episodes=50)
print(f"\nResults:")
print(f"  Average reward: {results['avg_reward']:.2f}")
print(f"  Max reward:     {results['max_reward']:.2f}")
print(f"  Min reward:     {results['min_reward']:.2f}")

# Save results
with open("../results/grpo_trained_rewards.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to results/grpo_trained_rewards.json")
