
# Flappy Bird Reinforcement Learning

A Python project implementing a **minimal version of Flappy Bird** with **reinforcement learning** (Q-Learning, SARSA, Expected SARSA, PPO). The project includes both a playable manual game and a `gymnasium`-compatible environment for training RL agents.

---

## Table of Contents

* [Installation](#installation)
* [Manual Game](#manual-game)
* [RL Environment](#rl-environment)
* [Tabular Agents](#tabular-agents)

  * [Q-Learning](#q-learning)
  * [SARSA](#sarsa)
  * [Expected SARSA](#expected-sarsa)
* [PPO with Stable-Baselines3](#ppo-with-stable-baselines3)
* [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
* [Code Overview](#code-overview)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/aratzBergado/ReinforcementLearning.git
cd https://github.com/aratzBergado/ReinforcementLearning.git
```

2. Install dependencies:

```bash
pip install pygame gymnasium stable-baselines3 torch optuna tensorboard numpy
```

3. Add project path to PYTHONPATH (if necessary):

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## Manual Game

To play Flappy Bird manually:

```bash
python flappyEnv/flappy.py
```

**Controls:**

* `Space` or left click: jump
* `R` or left click after Game Over: restart

---

## RL Environment

The project contains a `gymnasium`-compatible environment for training RL agents.

```python
from flappyEnv.flappy_env import FlappyEnv

env = FlappyEnv(render_mode="human")
obs, info = env.reset()
```

* **Observation:** `(bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y)`
* **Actions:** `0 = do nothing`, `1 = jump`
* **Reward:** positive for passing a pipe or staying near the center of the pipe, negative for collision.

---

## Tabular Agents

Includes classes for **Q-Learning**, **SARSA**, and **Expected SARSA** in `deustorl/`.

### Q-Learning

```python
from deustorl.qlearning import QLearning
from deustorl.common import EpsilonGreedyPolicy

agent = QLearning(env)
agent.learn(policy=EpsilonGreedyPolicy(0.1), n_steps=10000)
```

### SARSA

```python
from deustorl.sarsa import Sarsa
agent = Sarsa(env)
agent.learn(policy=EpsilonGreedyPolicy(0.1), n_steps=10000)
```

### Expected SARSA

```python
from deustorl.expected_sarsa import ExpectedSarsa
agent = ExpectedSarsa(env)
agent.learn(policy=EpsilonGreedyPolicy(0.1), n_steps=10000)
```

**Features:**

* Epsilon-greedy, softmax, random, weighted policies
* TensorBoard logging to visualize rewards and steps

---

## PPO with Stable-Baselines3

```python
from stable_baselines3 import PPO
from flappyEnv.flappy_env import FlappyEnv

env = FlappyEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_flappy")
```

### Evaluation

```python
from flappyEnv.evaluate import evaluate
evaluate("ppo_flappy", num_episodes=10, sleep_time=0.02)
```

---

## Hyperparameter Tuning with Optuna

Scripts are included to optimize PPO or tabular agent hyperparameters:

```bash
python optuna_ppo.py      # PPO optimization
python optuna_tabular.py  # SARSA, ExpectedSARSA, QLearning optimization
```

* Results saved in `optuna/`
* Plots generated: `optimization_history.html`, `param_importances.html`, etc.

---

## Code Overview

* `deustorl/` – Tabular algorithms and helpers (`common.py`, `helpers.py`...)
* `flappyEnv/` – RL environment (`FlappyEnv`, `flappy.py`, `flappy_for_curriculum.py`)
** `flappy.py` – Manual playable game using Pygame
* `training/` – Training scripts
* `optuna/optuna_analysis/` – Hyperparameter tuning scripts
* `evaluate.py` – Evaluation script for trained PPO models
* `model/` – Example of trained model

---

**Author:** Ilies Abdelsadok, Aratz Bergado, Natalia Gonzalez, Naroa Manterola

**Date:** 2026-01-07
