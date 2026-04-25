# encoder/behavioral_encoder.py
# Transformer-based behavioral signal encoder.
# Processes 7-day behavioral windows into latent representations.
# Makes system robust to daily noise.

import torch
import torch.nn as nn

class BehavioralSignalEncoder(nn.Module):
    """
    Temporal Transformer over 7-day behavioral windows.

    Architecture:
        Input: (batch, 7, input_dim) — 7 days of signals
        Transformer: 2 layers, 4 attention heads
        Output: (batch, output_dim) latent risk representation
        Risk head: 4-class hidden state inference
            (SAFE / VULNERABLE / AT_RISK / IN_DANGER)

    Why this matters:
        Rather than feeding raw daily numbers to the RL agent,
        the encoder learns to represent behavioral patterns in
        latent space. The agent reasons about patterns, not
        individual data points.
    """

    def __init__(self, input_dim=12, hidden_dim=64,
                 output_dim=32, n_heads=4):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Project raw signals to transformer dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Temporal transformer — attends across 7 days
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # Project to latent representation
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        # Risk inference head
        # Predicts hidden state probabilities from behavioral patterns
        self.risk_head = nn.Linear(output_dim, 4)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, behavioral_sequence):
        """
        behavioral_sequence: (batch, 7, input_dim)
        Returns:
            latent: (batch, output_dim) — for RL agent
            risk_probs: (batch, 4) — SAFE/VULN/AT_RISK/DANGER probs
        """
        x = self.input_projection(behavioral_sequence)
        x = self.transformer(x)
        x = x.mean(dim=1)  # pool across 7-day window
        latent = self.output_projection(x)
        risk_logits = self.risk_head(latent)
        risk_probs = self.softmax(risk_logits)
        return latent, risk_probs

    def encode_state(self, state_dict):
        """
        Converts a single state dictionary to tensor.
        Uses 12 core behavioral features.
        """
        features = [
            state_dict.get("activity_hour_mean", 20.0) / 26.0,
            state_dict.get("activity_hour_variance", 1.0) / 5.0,
            min(state_dict.get("unknown_contacts_today", 0), 10) / 10.0,
            min(state_dict.get("unknown_contact_message_volume", 0),
                120) / 120.0,
            (state_dict.get("friend_group_engagement_delta", 0.0)
             + 1.0) / 2.0,
            state_dict.get("family_response_rate", 0.75),
            (state_dict.get("sentiment_score", 0.0) + 1.0) / 2.0,
            (state_dict.get("sentiment_trend_7d", 0.0) + 1.0) / 2.0,
            state_dict.get("initiation_ratio", 0.5),
            state_dict.get("pursuit_score", 0.0),
            state_dict.get("late_night_conversation_rate", 0.0),
            state_dict.get("emotional_dependency_score", 0.0),
        ]
        return torch.tensor(features, dtype=torch.float32)

    def encode_window(self, state_window):
        """
        Encodes a 7-day window of states.
        state_window: list of 7 state dicts
        Returns: latent representation + risk probabilities
        """
        if len(state_window) < 7:
            # Pad with copies of first state if window not full yet
            padding = [state_window[0]] * (7 - len(state_window))
            state_window = padding + state_window

        tensors = [self.encode_state(s) for s in state_window]
        sequence = torch.stack(tensors).unsqueeze(0)  # (1, 7, input_dim)

        with torch.no_grad():
            latent, risk_probs = self.forward(sequence)

        return latent.squeeze(0), risk_probs.squeeze(0)

    def get_risk_label(self, state_window):
        """
        Returns predicted risk level as string.
        Used in demo to show encoder's inference.
        """
        _, risk_probs = self.encode_window(state_window)
        risk_idx = risk_probs.argmax().item()
        labels = ["SAFE", "VULNERABLE", "AT_RISK", "IN_DANGER"]
        confidence = risk_probs[risk_idx].item()
        return labels[risk_idx], round(confidence, 3)