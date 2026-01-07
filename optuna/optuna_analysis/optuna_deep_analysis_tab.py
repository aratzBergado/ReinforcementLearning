import os
import random
import json
import optuna
import numpy as np

from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning
from deustorl.helpers import DiscretizedObservationWrapper

from flappyEnv.flappy_env import FlappyEnv

from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
    plot_contour
)

def evaluate_tabular(env, q_table, max_policy, n_episodes=20):
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = max_policy(q_table, state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards)

def max_policy(q_table, state):
    return np.argmax(q_table[state])

def objective(trial):
    algo_name = trial.suggest_categorical("algo_name", ["sarsa", "esarsa", "qlearning"])

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-1, log=True)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.92, 0.9999)
    lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay", [100, 500, 1000, 5000])
    discount_rate = trial.suggest_float("discount_rate", 0.95, 0.999)
    epsilon = trial.suggest_float("epsilon", 0.01, 0.4)

    n_bins = trial.suggest_categorical("n_bins", [20, 30, 50, 80])

    n_steps = 500_000

    env = DiscretizedObservationWrapper(FlappyEnv(render_mode=None), n_bins=n_bins)

    if algo_name == "sarsa":
        algo = Sarsa(env)
    elif algo_name == "esarsa":
        algo = ExpectedSarsa(env)
    else:
        algo = QLearning(env)

    epsilon_policy = EpsilonGreedyPolicy(epsilon=epsilon)

    algo.learn(
        epsilon_policy,
        n_steps=n_steps,
        discount_rate=discount_rate,
        lr=lr,
        lrdecay=lr_decay,
        n_episodes_decay=lr_episodes_decay
    )

    avg_reward = evaluate_tabular(env, algo.q_table, max_policy, n_episodes=20)

    env.close()
    return avg_reward

if __name__ == "__main__":
    seed = 42
    random.seed(seed)

    os.makedirs("./optuna/FlappyTabularDeep/", exist_ok=True)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(
        direction='maximize',
        study_name="FlappyTabularDeep",
        storage="sqlite:///optuna/flappy_tabular_deep.db",
        load_if_exists=True,
        pruner=pruner
    )

    study.optimize(objective, n_trials=1)

    with open("optuna/FlappyTabularDeep/best_trial.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

    study = optuna.load_study(
        study_name="FlappyTabularDeep",
        storage="sqlite:///optuna/flappy_tabular_deep.db"
    )

    full_study_dir_path = "optuna/FlappyTabularDeep"
    os.makedirs(full_study_dir_path, exist_ok=True)

    plot_optimization_history(study).write_html(f"{full_study_dir_path}/optimization_history.html")
    plot_param_importances(study).write_html(f"{full_study_dir_path}/param_importances.html")
    plot_slice(study).write_html(f"{full_study_dir_path}/slice.html")
    plot_parallel_coordinate(study).write_html(f"{full_study_dir_path}/parallel.html")
    plot_contour(study).write_html(f"{full_study_dir_path}/contour.html")

    print("Tabular Study Phase A completed. Plots generated in optuna/FlappyTabularDeep/")
