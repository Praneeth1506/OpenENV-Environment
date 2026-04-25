# training/prompt_builder.py

def state_to_prompt(state):
    """
    Converts state dict to natural language prompt.
    Structured to encourage chain-of-thought reasoning
    before action selection — critical for GRPO.
    """
    
    archetype_descriptions = {
        "explorer": "naturally curious, high contact variance, new contacts are normal",
        "withdrawer": "naturally quiet, small friend group, deviation from baseline is key",
        "target": "initially normal baseline, rapid change is the primary signal",
    }
    
    return f"""You are SafeSignal, an AI system that helps parents stay aware 
of their child's digital wellbeing by observing behavioral patterns only.
You never read message content. Your only output is a recommendation 
for how the guardian should respond today.

=== CHILD PROFILE ===
Archetype: {state['child_archetype']}
Profile: {archetype_descriptions.get(state['child_archetype'], 'unknown')}

=== TODAY'S BEHAVIORAL SIGNALS ===
Activity timing: active around {state['activity_hour_mean']:.0f}:00
(late night = higher risk signal)

Contact patterns:
- Known friends contacted: {state['known_contacts_today']}
- Unknown contacts today: {state['unknown_contacts_today']}
- Messages with unknown contacts: {state['unknown_contact_message_volume']}
- Who initiates conversations: {state.get('initiation_ratio', 0.5):.0%} contact-initiated
- Contact pursuit score: {state.get('pursuit_score', 0.0):.2f} (1.0 = contact always re-initiates)

Social signals:
- Friend group engagement: {state['friend_group_engagement_delta']:+.0%} vs personal baseline
- Family response rate: {state['family_response_rate']:.0%}
- Emotional tone trend (7 days): {state['sentiment_trend_7d']:+.2f}

Relationship dynamics:
- Emotional dependency score: {state.get('emotional_dependency_score', 0.0):.2f}
- Late night conversation rate: {state.get('late_night_conversation_rate', 0.0):.0%}
- Migration readiness score: {state.get('migration_readiness_score', 0.0):.2f}

=== GUARDIAN CONTEXT ===
Trust level: {state['guardian_trust']:.0%}
Days since last action: {state['days_since_last_alert']}
Last guardian response: {state['last_alert_guardian_response']}
Consecutive ignored alerts: {state['consecutive_ignored_alerts']}

=== YOUR TASK ===
Think through this carefully before deciding.

Consider:
1. What do the behavioral signals suggest about risk level?
2. What does the child's archetype tell you about signal interpretation?
3. What does the guardian trust level tell you about intervention effectiveness?
4. What are the consequences of acting vs waiting?

Then choose exactly one action:
OBSERVE_QUIETLY — stay silent, preserve trust
GENTLE_AWARENESS — soft signal to guardian, early concern
PARENT_CHECK_IN — clear recommendation for conversation
URGENT_SUPPORT — direct high-urgency alert

Write your reasoning first, then state your action on the final line.
Action:"""


def parse_action(response_text):
    """
    Extracts action from model response.
    Searches for action keywords in priority order.
    """
    # Check for exact matches first (most reliable)
    actions_priority = [
        "URGENT_SUPPORT",
        "PARENT_CHECK_IN", 
        "GENTLE_AWARENESS",
        "OBSERVE_QUIETLY",
    ]
    
    response_upper = response_text.upper()
    
    for action in actions_priority:
        if action in response_upper:
            return action
    
    # Fuzzy matching fallback
    fuzzy_map = {
        "URGENT": "URGENT_SUPPORT",
        "SUPPORT": "URGENT_SUPPORT",
        "CHECK": "PARENT_CHECK_IN",
        "CHECK-IN": "PARENT_CHECK_IN",
        "GENTLE": "GENTLE_AWARENESS",
        "AWARENESS": "GENTLE_AWARENESS",
        "OBSERVE": "OBSERVE_QUIETLY",
        "SILENT": "OBSERVE_QUIETLY",
        "QUIETLY": "OBSERVE_QUIETLY",
        "WAIT": "OBSERVE_QUIETLY",
    }
    
    for keyword, action in fuzzy_map.items():
        if keyword in response_upper:
            return action
    
    # Default to safest option
    return "OBSERVE_QUIETLY"
