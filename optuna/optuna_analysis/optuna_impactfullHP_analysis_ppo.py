import os
import random
import json
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from flappyEnv.flappy_env import FlappyEnv
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_parallel_coordinate, plot_contour

def evaluate_sb3(model, env, n_eval_episodes=5):
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=False)
    return mean_reward

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    n_steps = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
    
    env = FlappyEnv(render_mode=None)
    model = PPO("MlpPolicy", env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, verbose=0)
    model.learn(total_timesteps=10_000)
    avg_reward = evaluate_sb3(model, env, n_eval_episodes=5)
    env.close()
    return avg_reward

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    os.makedirs("./optuna/", exist_ok=True)
    study = optuna.create_study(direction='maximize', study_name="FlappyPPO", storage="sqlite:///optuna/flappy_ppo.db", load_if_exists=True)
    study.optimize(objective, n_trials=1)

    with open("optuna/FlappyPPO/best_trial.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)

study = optuna.load_study(study_name="FlappyPPO", storage="sqlite:///optuna/flappy_ppo.db")

full_study_dir_path = "optuna/FlappyPPO"
os.makedirs(full_study_dir_path, exist_ok=True)

plot_optimization_history(study).write_html(f"{full_study_dir_path}/optimization_history.html")
plot_param_importances(study).write_html(f"{full_study_dir_path}/param_importances.html")
plot_slice(study).write_html(f"{full_study_dir_path}/slice.html")
plot_parallel_coordinate(study).write_html(f"{full_study_dir_path}/parallel.html")
plot_contour(study).write_html(f"{full_study_dir_path}/contour.html")

print("Plots générés. Vérifie le dossier optuna/FlappyPPO/")
