# Approximating Perfection: An RL Approach to Video Poker

This project explores the application of Reinforcement Learning (RL) to the game of Video Poker. Video Poker presents a unique challenge: it is a game of chance, but it also has a mathematically solvable optimal strategy. This makes it an ideal environment to benchmark the performance of an RL agent against a perfect player.

The core of this project is a comparison between two agents:

1.  **An Optimal Agent:** This agent uses a pre-computed, brute-force solution that represents the perfect, mathematically optimal strategy for every possible hand. It serves as the "ground truth" for the best possible performance.

2.  **A Deep Q-Network (DQN) Agent:** This is a reinforcement learning agent that learns to play the game from scratch, with no prior knowledge of the rules or strategy. Its goal is to learn a policy that maximizes its score through trial and error.

By training the DQN agent and comparing its performance to the optimal agent, this project aims to answer the question: **How close can a model-free reinforcement learning agent get to the perfect strategy in a complex, solvable game?**

This repository allows for training the DQN agent, evaluating both agents, and even playing the game yourself to see how your decisions stack up against an AI or a Perfect Agent.

## Setup

1. Create and activate conda environment:
```bash
conda create -n MLSec_v0 python=3.8
conda activate MLSec_v0
```

2. Install dependencies:
```bash
pip install torch numpy matplotlib tqdm gym tensorboard
```

## Usage Guide

### 1. Play as Human

You can play Video Poker interactively:

```bash
python video_poker.py
```

The game will:
- Deal 5 cards
- Let you choose which cards to hold/discard
- Show the optimal play for comparison
- Calculate your final hand value and payout

### 2. Train a DQN Agent

Train a Deep Q-Network agent with customizable parameters:

```bash
python train_agent.py \
    --episodes 2000 \
    --max-steps 100 \
    --eps-start 1.0 \
    --eps-end 0.01 \
    --eps-decay 0.995 \
    --checkpoint-freq 1000 \
    --model-dir models \
    --log-dir runs/video_poker \
    --decay-type exponential \
    --decay-percent 80
```

Key parameters:
- `--episodes`: Total training episodes
- `--max-steps`: Maximum steps per episode
- `--eps-start`: Starting exploration rate
- `--eps-end`: Minimum exploration rate
- `--decay-type`: Choose between 'exponential' or 'linear' epsilon decay
- `--decay-percent`: For linear decay, percentage of episodes over which to decay epsilon

Monitor training with TensorBoard:
```bash
tensorboard --logdir=runs/video_poker
```

The training process automatically saves:
- Regular checkpoints (based on `--checkpoint-freq`)
- The best model (based on 100-episode moving average score)
- The final model

### 3. Evaluate the Optimal (Brute Force) Agent

The optimal agent uses a pre-computed solution dictionary to make perfect decisions:

```bash
python play_with_optimal_agent.py --num-games 10000
```

This will:
- Run the optimal agent for the specified number of games
- Generate a histogram of scores
- Calculate mean score and standard deviation
- The solution dictionary is cached for faster loading in subsequent runs

### 4. Evaluate a Trained DQN Agent

Evaluate your trained DQN agent in two modes:

```bash
# Interactive mode - play and see agent decisions
python play_with_dqn_agent.py --model best_model.pth --mode interactive

# Compare mode - compare with optimal agent
python play_with_dqn_agent.py --model best_model.pth --mode compare --num-games 10000
```

The comparison will show:
- Performance statistics for both agents
- Differences in decision-making
- Overall score comparison

### 5. Interactive Play with Both Agents

You can play interactively while seeing recommendations from both the DQN and optimal agents:

```bash
python play_with_dqn_agent.py --model best_model.pth --mode interactive
```

This allows you to:
- See your cards and make decisions
- View what the DQN agent would do
- View what the optimal agent would do

## Project Structure

- `poker_classes.py`: Core poker game logic (cards, hands, deck)
- `video_poker.py`: Main game implementation with human play interface
- `poker_env.py`: Gym environment wrapper for reinforcement learning
- `dqn_agent.py`: Deep Q-Network implementation with experience replay
- `train_agent.py`: Training script with TensorBoard logging
- `play_with_optimal_agent.py`: Evaluation script for the optimal agent
- `play_with_dqn_agent.py`: Evaluation script for the DQN agent
- `solution_loader.py`: Optimized loader for the brute-force solutions
- `brute_force_solving.py`: Script to generate optimal solutions

## Performance Comparison

The DQN agent learns to approximate the optimal strategy through experience. While it may not match the perfect play of the brute-force agent, it demonstrates how reinforcement learning can be applied to complex decision problems.

The optimal agent achieves a mean score of approximately 26 points per game, which represents the theoretical maximum expected value for this variant of Video Poker.

The DQN agent, after training, achieves a mean score of approximately 22 points per game, which is lower than the optimal agent but still demonstrates the power of reinforcement learning in approximating optimal strategies.

The folder `media` contains the following files:
- `OptimalAgent_rewards_distribution.png`: Histogram of scores for the optimal agent
- `DQN_rewards_distribution.png`: Histogram of scores for the DQN agent

![Optimal Agent Rewards Distribution](media/OptimalAgent_rewards_distribution.png)
![DQN Agent Rewards Distribution](media/DQN_rewards_distribution.png)