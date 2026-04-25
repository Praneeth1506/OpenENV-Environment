---
title: SafeSignal
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
---

# SafeSignal

An RL training environment that teaches an AI agent to detect shifts 
in a child's online behavioral patterns and decide — with the precision 
of a thoughtful parent — when to alert a guardian, when to stay silent, 
and how to preserve the trust that makes future warnings matter.

## Structure
- `environment/` — OpenEnv simulation, child archetypes, reward function
- `training/` — PPO training pipeline
- `demo/` — Gradio demo interface

## Run locally
pip install -r requirements.txt
py app.py

## Tech stack
OpenEnv · HuggingFace TRL · Llama 3.2 1B · Gradio · Plotly