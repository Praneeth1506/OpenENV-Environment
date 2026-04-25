import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'environment'))
from safesignal_env import SafeSignalEnv
from prompt_builder import state_to_prompt, parse_action
import json, numpy as np, torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from unsloth import FastLanguageModel
    UNSLOTH = True
except ImportError:
    UNSLOTH = False
    print("ℹ️ Note: unsloth not found (expected on Mac). Using standard transformers.")

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

import trl
from trl import PPOConfig
TRL_NEW_API = int(trl.__version__.split(".")[0]) >= 1 or int(trl.__version__.split(".")[1]) >= 9

if TRL_NEW_API:
    try:
        from trl import PPOv2Trainer as PPOTrainer
    except ImportError:
        from trl import PPOTrainer
else:
    from trl import PPOTrainer

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ungated, use if 403
MAX_SEQ_LEN = 512
MAX_NEW_TOKENS = 80
N_EPISODES = 2000
CHECKPOINT_EVERY = 500
LOG_EVERY = 2
VALID_ACTIONS = ["OBSERVE_QUIETLY", "GENTLE_AWARENESS",
                 "PARENT_CHECK_IN", "URGENT_SUPPORT"]

def load_model(model_name):
    if UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16, lora_dropout=0,
            bias="none", use_gradient_checkpointing=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none"
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name, 
            peft_config=lora_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

try:
    model, tokenizer = load_model(MODEL_NAME)
    print(f"Loaded {MODEL_NAME}")
except Exception:
    print(f"ℹ️ Note: {MODEL_NAME} requires HuggingFace authentication.")
    print(f"Using fallback model: {FALLBACK_MODEL}")
    model, tokenizer = load_model(FALLBACK_MODEL)
    print(f"Loaded fallback {FALLBACK_MODEL}")

if TRL_NEW_API:
    # TRL >= 0.9: PPOConfig takes different args, step() takes different inputs
    ppo_config = PPOConfig(
        learning_rate=1.5e-5,
        batch_size=16,
        mini_batch_size=16,
        ppo_epochs=4,
        gamma=0.99,
    )
    trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    def ppo_step(trainer, query_tensors, response_tensors, rewards):
        # New API: step takes lists of 1D tensors, one per sample
        # Pad queries and responses to uniform length within the batch
        q_padded = torch.nn.utils.rnn.pad_sequence(
            query_tensors, batch_first=True,
            padding_value=tokenizer.pad_token_id)
        r_padded = torch.nn.utils.rnn.pad_sequence(
            response_tensors, batch_first=True,
            padding_value=tokenizer.pad_token_id)
        reward_tensors = [r if r.dim() > 0 else r.unsqueeze(0) for r in rewards]
        trainer.step(
            list(q_padded), list(r_padded), reward_tensors
        )
else:
    # TRL < 0.9: original API
    ppo_config = PPOConfig(
        learning_rate=1.5e-5,
        batch_size=10,
        mini_batch_size=10,
        ppo_epochs=4,
        gamma=0.99,
        log_with="wandb" if WANDB else None,
    )
    trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    def ppo_step(trainer, query_tensors, response_tensors, rewards):
        reward_tensors = [r.unsqueeze(0) if r.dim() == 0 else r for r in rewards]
        trainer.step(query_tensors, response_tensors, reward_tensors)

def generate_response(model, tokenizer, prompt):
    """Returns (response_text, response_ids_1d_tensor). Never raises."""
    try:
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN,
            padding=False,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        # Ensure correct device
        input_ids = input_ids.to(next(model.parameters()).device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
            
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                max_length=None,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Slice off the prompt tokens — keep only generated tokens
        response_ids = output_ids[0][input_ids.shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text, response_ids.cpu()
    except Exception as e:
        print(f"  [generate] Error: {e} — using default action")
        fallback = "Action: OBSERVE_QUIETLY"
        fallback_ids = tokenizer(
            fallback, return_tensors="pt")["input_ids"].squeeze(0).cpu()
        return fallback, fallback_ids

env = SafeSignalEnv()
episode_rewards = []
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

for episode in range(N_EPISODES):
    state = env.reset()
    done = False

    query_tensors = []
    response_tensors = []
    reward_list = []

    while not done:
        prompt = state_to_prompt(state)
        response_text, response_ids = generate_response(model, tokenizer, prompt)
        action = parse_action(response_text)
        state, reward, done, info = env.step(action)

        # Tokenize prompt for PPO (1D tensor, no batch dim)
        query_ids = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN,
            padding=False,
        )["input_ids"].squeeze(0).cpu()

        query_tensors.append(query_ids)
        response_tensors.append(response_ids)
        # reward is a plain float from env.step — wrap in scalar tensor
        reward_list.append(torch.tensor(float(reward)))

    # One PPO update per complete 30-step episode
    try:
        bs = ppo_config.batch_size
        ppo_step(trainer, query_tensors[-bs:], response_tensors[-bs:], reward_list[-bs:])
    except Exception as e:
        print(f"  [ppo_step] Episode {episode}: {e} — skipping update")

    episode_total = sum(r.item() for r in reward_list)
    episode_rewards.append(round(episode_total, 4))

    if (episode + 1) % LOG_EVERY == 0:
        recent = np.mean(episode_rewards[-LOG_EVERY:])
        print(f"Episode {episode+1}/{N_EPISODES} "
              f"— mean reward (last {LOG_EVERY}): {recent:.3f} "
              f"— final hidden state: {info['hidden_state']} "
              f"— guardian trust: {info['guardian_trust']:.2f}")

    if (episode + 1) % CHECKPOINT_EVERY == 0:
        ckpt = Path(__file__).parent / "checkpoints" / f"episode_{episode+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt))
        tokenizer.save_pretrained(str(ckpt))
        print(f"  Checkpoint saved → {ckpt}")

out = {
    "episode_rewards": episode_rewards,
    "mean": round(float(np.mean(episode_rewards)), 4),
    "std": round(float(np.std(episode_rewards)), 4),
    "min": round(float(np.min(episode_rewards)), 4),
    "max": round(float(np.max(episode_rewards)), 4),
    "config": {
        "n_episodes": N_EPISODES,
        "agent": "ppo",
        "model": MODEL_NAME,
        "gamma": 0.99,
        "episode_length": 30,
        "trl_new_api": TRL_NEW_API,
    }
}
with open(results_dir / "trained_rewards.json", "w") as f:
    json.dump(out, f, indent=2)
print("Done. Results → results/trained_rewards.json")
