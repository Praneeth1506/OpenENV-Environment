# training/train_grpo.py
# Main GRPO training script — A100 / CUDA optimised.
# Connects to SafeSignalEnvironment in real time — not a static dataset.

import sys
import os
import json
import random
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ungated, use if 403
MAX_SEQ_LEN      = 512
MAX_NEW_TOKENS   = 80
N_EPISODES       = 200
CHECKPOINT_EVERY = 50
LOG_EVERY        = 2
VALID_ACTIONS    = ["OBSERVE_QUIETLY", "GENTLE_AWARENESS",
                    "PARENT_CHECK_IN", "URGENT_SUPPORT"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = torch.cuda.is_available()   # 4-bit quant only on GPU
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

print(f"Device: {DEVICE}  |  4-bit: {USE_4BIT}  |  bf16: {USE_BF16}")

from server.environment import SafeSignalEnvironment, ACTIONS
from training.prompt_builder import state_to_prompt, parse_action
from training.grpo_rewards import compute_grpo_reward

# ── Step 1: Load Model ─────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)

    kwargs = dict(low_cpu_mem_usage=True)
    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32

    try:
        kwargs["attn_implementation"] = "flash_attention_2"
        m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        m = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    return m, tokenizer

print(f"\nLoading {MODEL_NAME} ...")
try:
    model, tokenizer = load_model(MODEL_NAME)
except Exception as e:
    print(f"  Primary failed ({e}), loading fallback: {FALLBACK_MODEL}")
    model, tokenizer = load_model(FALLBACK_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Step 2: Build Training Dataset ─────────────────────────────────────────
print(f"\nGenerating training prompts from {N_EPISODES} live environment episodes...")

training_samples = []
env = SafeSignalEnvironment()

for episode in range(N_EPISODES):
    result = env.reset()
    state  = result.observation
    done   = False

    while not done:
        prompt = state_to_prompt(state)
        training_samples.append({
            "prompt":        prompt,
            "hidden_state":  env.child.hidden_state,
            "state_snapshot": dict(state),
        })
        action = random.choice(ACTIONS)
        step   = env.step(action)
        state  = step.observation
        done   = step.done

print(f"Generated {len(training_samples)} training prompts")

training_metadata = {
    i: {
        "hidden_state":   training_samples[i]["hidden_state"],
        "state_snapshot": training_samples[i]["state_snapshot"],
    }
    for i in range(len(training_samples))
}

from datasets import Dataset
dataset = Dataset.from_list([
    {"prompt": s["prompt"]} for s in training_samples
])

# ── Step 3: Reward Function for GRPO ───────────────────────────────────────
sample_index = [0]


def grpo_reward_fn(prompts, completions, **kwargs):
    rewards = []

    for prompt, completion in zip(prompts, completions):
        idx  = sample_index[0] % len(training_metadata)
        meta = training_metadata[idx]

        response_text = (
            completion[0]["content"]
            if isinstance(completion, list)
            else completion
        )

        action       = parse_action(response_text)
        hidden_state = meta["hidden_state"]
        state        = meta["state_snapshot"]

        temp_env = SafeSignalEnvironment(
            archetype=state.get("child_archetype", "target")
        )
        temp_env.reset()
        temp_env.child.guardian_trust = state.get("guardian_trust", 0.8)
        temp_env.child.consecutive_ignored_alerts = state.get(
            "consecutive_ignored_alerts", 0
        )
        guardian_response = temp_env.child.simulate_guardian_response(action)

        reward = compute_grpo_reward(
            response_text    = response_text,
            state            = state,
            hidden_state     = hidden_state,
            guardian_response= guardian_response,
        )
        rewards.append(float(reward))

    sample_index[0] += 1
    return rewards


# ── Step 4: GRPO Configuration ─────────────────────────────────────────────
import inspect
import trl

GRPOConfig = getattr(trl, "GRPOConfig", None)
GRPOTrainer = getattr(trl, "GRPOTrainer", None)
if GRPOConfig is None or GRPOTrainer is None:
    from trl import ORPOConfig as GRPOConfig, ORPOTrainer as GRPOTrainer
    print("Warning: GRPO classes not found in TRL; falling back to ORPO classes.")

raw_config_kwargs = {
    "num_generations":             8,
    "max_prompt_length":           MAX_SEQ_LEN,
    "max_completion_length":       MAX_NEW_TOKENS,
    "learning_rate":               5e-6,
    "per_device_train_batch_size": 4 if DEVICE == "cuda" else 1,
    "gradient_accumulation_steps": 2 if DEVICE == "cuda" else 8,
    "num_train_epochs":            1,
    "beta":                        0.01,
    "output_dir":                  "./training/checkpoints",
    "logging_steps":               LOG_EVERY,
    "save_steps":                  CHECKPOINT_EVERY,
    "warmup_ratio":                0.1,
    "lr_scheduler_type":           "cosine",
    "bf16":                        USE_BF16,
    "fp16":                        (DEVICE == "cuda" and not USE_BF16),
    "dataloader_num_workers":      4 if DEVICE == "cuda" else 0,
    "report_to":                   "none",
}

allowed_args = set(inspect.signature(GRPOConfig.__init__).parameters) - {"self"}
config_kwargs = {k: v for k, v in raw_config_kwargs.items() if k in allowed_args}
unsupported = set(raw_config_kwargs) - allowed_args
if unsupported:
    print("Filtered unsupported GRPO/ORPO config args:", unsupported)

grpo_config = GRPOConfig(**config_kwargs)

# ── Step 5: Train ───────────────────────────────────────────────────────────
trainer = GRPOTrainer(
    model         = model,
    reward_funcs  = grpo_reward_fn,
    args          = grpo_config,
    train_dataset = dataset,
    processing_class = tokenizer,
)

print(f"\nStarting GRPO training...")
print(f"  Model:                {MODEL_NAME}")
print(f"  Device:               {DEVICE}")
print(f"  Training episodes:    {N_EPISODES}")
print(f"  Training prompts:     {len(dataset)}")
print(f"  Responses per prompt: {grpo_config.num_generations}")
print(f"  Batch size:           {grpo_config.per_device_train_batch_size}")
print(f"  Checkpoint every:     {CHECKPOINT_EVERY} steps")
print(f"  Log every:            {LOG_EVERY} steps")
print()

trainer.train()
trainer.save_model("./training/checkpoints/final")
print("\nTraining complete. Model saved.")

# ── Step 6: Evaluate Trained Model ─────────────────────────────────────────
print("\nEvaluating trained model...")
model.eval()

eval_results   = []
eval_trust     = []
eval_outcomes  = []
rubric_history = []

for episode in range(100):
    env    = SafeSignalEnvironment()
    result = env.reset()
    state  = result.observation
    done   = False
    ep_reward  = 0
    ep_rubrics = []

    while not done:
        prompt = state_to_prompt(state)
        inputs = tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = MAX_SEQ_LEN,
        ).to(DEVICE if not USE_4BIT else model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                temperature    = 0.1,
                do_sample      = True,
                pad_token_id   = tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        action = parse_action(response)
        step   = env.step(action)
        ep_reward += step.reward

        if step.rubric_scores:
            ep_rubrics.append(step.rubric_scores)

        state = step.observation
        done  = step.done

    eval_results.append(round(ep_reward, 3))
    eval_trust.append(round(step.info["guardian_trust"], 3))
    eval_outcomes.append(step.info["hidden_state"])

    if ep_rubrics:
        avg_rubric = {}
        for name in ["intervention_timing", "guardian_trust",
                     "silence_intelligence"]:
            scores = [
                r.get(name, {}).get("weighted_score", 0)
                for r in ep_rubrics if name in r
            ]
            if scores:
                avg_rubric[name] = sum(scores) / len(scores)
        rubric_history.append(avg_rubric)

    if episode % 20 == 0:
        avg = sum(eval_results) / len(eval_results)
        print(f"  Episode {episode:3d} | Avg reward: {avg:+.2f}")

trained_avg      = sum(eval_results) / len(eval_results)
trained_safe_pct = (
    sum(1 for o in eval_outcomes if o == "SAFE") / len(eval_outcomes) * 100
)

print(f"\n  Trained avg reward:  {trained_avg:+.2f}")
print(f"  % ended safe:        {trained_safe_pct:.1f}%")

os.makedirs("results", exist_ok=True)
with open("results/trained_rewards.json", "w") as f:
    json.dump({
        "policy":          "grpo_trained",
        "n_episodes":      100,
        "avg_reward":      round(trained_avg, 3),
        "pct_ended_safe":  round(trained_safe_pct, 1),
        "episode_rewards": eval_results,
        "trust_trajectory": eval_trust,
        "outcomes":        eval_outcomes,
        "rubric_history":  rubric_history,
    }, f, indent=2)
print("Saved: results/trained_rewards.json")

# ── Step 7: Push model to HuggingFace Hub (if HF_TOKEN is set) ─────────────
hf_token = os.environ.get("HF_TOKEN")
hf_repo  = os.environ.get("HF_REPO", "safesignal-grpo-tinyllama")

if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    hf_username = api.whoami()["name"]
    repo_id = f"{hf_username}/{hf_repo}"
    print(f"\nPushing model to Hub: {repo_id}")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path="./training/checkpoints/final",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj="results/trained_rewards.json",
            path_in_repo="trained_rewards.json",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Model saved to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Hub push failed: {e} — model is still at ./training/checkpoints/final")
else:
    print("\nHF_TOKEN not set — skipping Hub push.")
    print("Model is at: ./training/checkpoints/final")

print("\nDone.")
