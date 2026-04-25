---
title: SafeSignal
emoji: 🚦
colorFrom: indigo
colorTo: blue
sdk: python
python_version: "3.11"
app_file: space_entrypoint.py
pinned: false
---

# OpenENV-Environment

A reinforcement learning environment for training AI models to make safe and appropriate decisions in child safety scenarios. This project implements both PPO and GRPO training approaches using the SafeSignal environment.

## 🚀 Live Demo

Experience the trained model in action: [Hugging Face Space](https://huggingface.co/spaces/Deni1315/SafeSignal)

## 📋 Overview

This project simulates a guardian AI system that must respond appropriately to various child safety situations. The AI learns to balance:
- **Safety**: Prioritizing urgent situations requiring immediate intervention
- **Trust**: Building and maintaining trust with the child through appropriate responses
- **Development**: Allowing natural child development without over-intervention

The environment includes realistic scenarios like tantrums, injuries, emotional distress, and developmental milestones.

## 🏗️ Project Structure

```
├── demo/                          # Demo applications
│   ├── demo_app.py               # Interactive demo
│   ├── demo_scenarios.py         # Pre-defined scenarios
│   └── visualizer.py             # Results visualization
├── environment/                  # RL environment code
│   ├── safesignal_env.py         # Main environment
│   ├── simulated_child.py        # Child behavior simulation
│   ├── reward.py                 # Reward calculation
│   ├── rubrics.py                # Safety evaluation rubrics
│   └── constants.py              # Environment constants
├── training/                     # Training scripts and utilities
│   ├── train.py                  # PPO training script
│   ├── train_grpo.py             # GRPO training script
│   ├── evaluate.py               # Model evaluation
│   ├── prompt_builder.py         # Text prompt generation
│   ├── grpo_rewards.py           # GRPO reward functions
│   ├── baseline.py               # Baseline evaluation
│   └── plots.py                  # Training visualization
├── results/                      # Training results and plots
├── trained_model/                # Saved model weights
└── requirements.txt              # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Praneeth1506/OpenENV-Environment.git
   cd OpenENV-Environment
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Hugging Face authentication (optional, for gated models):**
   ```bash
   huggingface-cli login
   ```

## 🎯 Usage

### Training

#### PPO Training (Default)
```bash
python training/train.py
```
Trains for 200 episodes using Proximal Policy Optimization.

#### GRPO Training
```bash
python training/train_grpo.py
```
Trains for 200 episodes using Group Relative Policy Optimization.

### Evaluation

#### Evaluate a trained model:
```bash
python training/evaluate.py
```

#### Run baseline evaluation:
```bash
python training/baseline.py
```

### Demo

#### Interactive demo:
```bash
python demo/demo_app.py
```

#### Run specific scenarios:
```bash
python demo/demo_scenarios.py
```

#### Visualize results:
```bash
python demo/visualizer.py
```

## 🤖 Model Details

- **Base Model**: TinyLlama-1.1B-Chat-v1.0 (fallback to ungated version if needed)
- **Fine-tuning**: LoRA with rank 16
- **Training Episodes**: 200 (configurable)
- **Episode Length**: 30 steps
- **Actions**: 4 discrete actions (OBSERVE_QUIETLY, GENTLE_AWARENESS, PARENT_CHECK_IN, URGENT_SUPPORT)

## 📊 Environment Dynamics

The SafeSignal environment simulates:
- **Child States**: Emotional state, physical condition, developmental stage
- **Guardian Actions**: Different intervention levels
- **Rewards**: Based on safety outcomes, trust maintenance, and appropriate responses
- **Scenarios**: Tantrums, injuries, emotional needs, developmental milestones

## 🔬 Research Aspects

This project explores:
- **Safe AI Decision Making**: Balancing multiple objectives in safety-critical scenarios
- **Trust-Aware AI**: Understanding how AI actions affect human trust
- **Reinforcement Learning**: Comparing PPO vs GRPO approaches
- **Child Development**: Appropriate AI responses to different developmental stages

## 📈 Results

Training results are saved to `results/` directory including:
- Episode reward curves
- Trust vs safety trade-offs
- Rubric breakdown analysis
- Model checkpoints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Transformers](https://huggingface.co/docs/transformers/index) and [TRL](https://huggingface.co/docs/trl/index)
- Environment inspired by child safety research
- Deployed on [Hugging Face Spaces](https://huggingface.co/spaces)

## 📞 Contact

For questions or collaborations, please open an issue on GitHub.