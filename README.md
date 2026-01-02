# Flappy Bird Reinforcement Learning

Un projet Python implémentant un **Flappy Bird minimal** avec **apprentissage par renforcement** (Q-Learning, SARSA, Expected SARSA, PPO). Le projet inclut à la fois un jeu jouable manuellement et un environnement compatible avec `gymnasium` pour entraîner des agents.

---

## Table des matières

* [Installation](#installation)
* [Jeu classique](#jeu-classique)
* [Environnement RL](#environnement-rl)
* [Agents tabulaires](#agents-tabulaires)

  * [Q-Learning](#q-learning)
  * [SARSA](#sarsa)
  * [Expected SARSA](#expected-sarsa)
* [PPO avec Stable-Baselines3](#ppo-avec-stable-baselines3)
* [Hyperparameter tuning avec Optuna](#hyperparameter-tuning-avec-optuna)
* [Fonctionnalités](#fonctionnalités)
* [Aperçu du code](#aperçu-du-code)

---

## Installation

1. Cloner le projet :

```bash
git clone <ton_repo>
cd <ton_repo>
```

2. Installer les dépendances :

```bash
pip install pygame gymnasium stable-baselines3 torch optuna tensorboard numpy
```

3. Ajouter le chemin du projet à PYTHONPATH (si nécessaire) :

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## Jeu classique

Pour jouer au Flappy Bird manuellement :

```bash
python flappy_bird.py
```

**Contrôles :**

* `Espace` ou clic gauche : sauter
* `R` ou clic gauche après Game Over : redémarrer

---

## Environnement RL

Le projet contient un environnement `gymnasium` compatible pour entraîner des agents RL.

```python
from flappyEnv.flappy_env import FlappyEnv
env = FlappyEnv(render_mode="human")
obs, info = env.reset()
```

* Observation : `(bird_y, bird_velocity, next_pipe_x, next_pipe_gap_y)`
* Actions : `0 = rien`, `1 = jump`
* Reward : positif pour passer un pipe ou se rapprocher du centre du pipe, négatif en cas de collision.

---

## Agents tabulaires

Les classes pour **Q-Learning**, **SARSA** et **Expected SARSA** sont fournies dans le dossier `deustorl`.

### Q-Learning

```python
from deustorl.qlearning import QLearning
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

**Fonctionnalités :**

* Politique epsilon-greedy, softmax, random, weighted
* Enregistrement TensorBoard pour visualiser les récompenses et les étapes

---

## PPO avec Stable-Baselines3

```python
from stable_baselines3 import PPO
from flappyEnv.flappy_env import FlappyEnv

env = FlappyEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("ppo_flappy")
```

### Évaluation

```python
from flappyEnv.evaluate import evaluate
evaluate("ppo_flappy", num_episodes=10, sleep_time=0.02)
```

---

## Hyperparameter tuning avec Optuna

Le projet contient un script pour optimiser les hyperparamètres PPO ou tabulaire :

```bash
python optuna_ppo.py   # PPO
python optuna_tabular.py  # SARSA, ExpectedSARSA, QLearning
```

* Les résultats sont sauvegardés dans `optuna/`
* Plots générés : `optimization_history.html`, `param_importances.html`, etc.

---

## Fonctionnalités

* Flappy Bird minimal jouable et modifiable
* Compatible `gymnasium` pour RL
* Agents tabulaires : Q-Learning, SARSA, Expected SARSA
* Agent profond : PPO
* Support TensorBoard pour monitoring
* Wrapper pour discrétiser les observations et actions continues
* Optimisation hyperparamètres avec Optuna
* Évaluation automatique d’agents

---

## Aperçu du code

* `flappy_bird.py` : jeu classique avec Pygame
* `flappyEnv/` : environnement RL (`FlappyEnv`, `flappy.py`)
* `deustorl/` : algorithmes tabulaires et helpers (`common.py`, `helpers.py`)
* `optuna_*.py` : scripts d’optimisation hyperparamètres
* `evaluate.py` : script pour tester un modèle PPO entraîné

---

**Auteur** : Ilies Abdelsadok, 
**Date** : 2026-01-02
**Licence** : MIT

---

Si tu veux, je peux te faire une **version encore plus graphique avec diagrammes et schémas** qui explique les classes `QTable`, `Sarsa`, et `FlappyEnv` dans le README. Cela rendrait le projet beaucoup plus lisible pour quelqu’un qui découvre le code.

Veux‑tu que je fasse ça ?
