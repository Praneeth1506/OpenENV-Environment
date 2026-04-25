# training/grpo_rewards.py

from prompt_builder import parse_action

def compute_grpo_reward(response_text, state, hidden_state, 
                         guardian_response, episode_history):
    """
    Complete reward function for GRPO training.
    Three components scored separately then combined.
    """
    
    reward = 0.0
    
    # ── Component 1: Action Correctness Reward ─────────────────────────
    # Did the agent choose the right action given the true hidden state?
    action = parse_action(response_text)
    action_reward = compute_action_reward(action, state, hidden_state, 
                                          guardian_response)
    reward += action_reward
    
    # ── Component 2: Reasoning Quality Reward ──────────────────────────
    # Did the agent show good reasoning before choosing the action?
    # This is what makes GRPO better than PPO for your problem —
    # you can reward the quality of reasoning, not just the outcome.
    reasoning_reward = compute_reasoning_reward(response_text, state, 
                                                 hidden_state)
    reward += reasoning_reward * 0.3  # weighted lower than outcome
    
    # ── Component 3: Format Compliance Reward ──────────────────────────
    # Did the agent follow the required output format?
    format_reward = compute_format_reward(response_text)
    reward += format_reward * 0.1
    
    return reward


def compute_action_reward(action, state, hidden_state, guardian_response):
    """Same reward logic as your existing reward.py"""
    from reward import compute_immediate_reward
    return compute_immediate_reward(action, state, hidden_state, 
                                    guardian_response)


def compute_reasoning_reward(response_text, state, hidden_state):
    """
    Rewards the agent for showing correct reasoning.
    This is the key GRPO addition — you reward the thought process,
    not just the conclusion.
    """
    reward = 0.0
    text_lower = response_text.lower()
    
    # Reward for mentioning trust when trust is low
    if state["guardian_trust"] < 0.4:
        if "trust" in text_lower:
            reward += 0.3  # agent acknowledged the trust constraint
        else:
            reward -= 0.2  # agent ignored the most important constraint
    
    # Reward for mentioning timing when alert was recent
    if state["days_since_last_alert"] < 3:
        if any(word in text_lower for word in 
               ["recent", "soon", "wait", "timing", "frequency"]):
            reward += 0.2
    
    # Reward for mentioning archetype in reasoning
    archetype = state["child_archetype"]
    if archetype in text_lower:
        reward += 0.2  # agent considered child's personality
    
    # Reward for mentioning silence as valid option
    if hidden_state in ["SAFE", "VULNERABLE"]:
        if any(word in text_lower for word in 
               ["wait", "observe", "silent", "monitor", "watch"]):
            reward += 0.2
    
    # Penalize overconfident language when situation is ambiguous
    if hidden_state == "VULNERABLE":
        if any(word in text_lower for word in 
               ["definitely", "certainly", "must", "urgent"]):
            reward -= 0.3
    
    # Reward for explicitly weighing tradeoffs
    if any(phrase in text_lower for phrase in 
           ["however", "but", "on the other hand", "tradeoff", 
            "balance", "risk of"]):
        reward += 0.15  # agent is reasoning, not just reacting
    
    return reward


def compute_format_reward(response_text):
    """
    Rewards correct output format.
    Agent must state action clearly on final line.
    """
    valid_actions = [
        "OBSERVE_QUIETLY", "GENTLE_AWARENESS", 
        "PARENT_CHECK_IN", "URGENT_SUPPORT"
    ]
    
    # Check action appears in response
    action_found = any(action in response_text.upper() 
                       for action in valid_actions)
    if not action_found:
        return -0.5  # penalize missing action
    
    # Check reasoning exists before action
    lines = [l.strip() for l in response_text.strip().split('\n') 
             if l.strip()]
    if len(lines) < 2:
        return -0.3  # penalize no reasoning, just action
    
    return 0.5  # correct format
