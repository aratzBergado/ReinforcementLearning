import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from flappy_env import FlappyEnv

MODEL_PATH = "ppo_flappy"

BEST_PARAMS = {
    "learning_rate": 1.1445998894919558e-05,
    "gamma": 0.9087886548031125,
    "n_steps": 128
}

TOTAL_TIMESTEPS = 1_000_000

train_env = FlappyEnv(render_mode=None, frame_skip=3)

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="ppo_flappy",
)

# Charger ou créer le modèle
if os.path.exists(MODEL_PATH + ".zip"):
    print("existing model loading...")
    model = PPO.load(MODEL_PATH, env=train_env)
    start_timesteps = model.num_timesteps
else:
    print("no existing model, creating new...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=BEST_PARAMS["learning_rate"],
        gamma=BEST_PARAMS["gamma"],
        n_steps=BEST_PARAMS["n_steps"],
        batch_size=BEST_PARAMS["n_steps"],
        ent_coef=0.0,
        gae_lambda=0.95,
        clip_range=0.2
    )
    start_timesteps = 0

# Calculer le nombre de timesteps restant à faire
remaining_timesteps = max(0, TOTAL_TIMESTEPS - start_timesteps)

if remaining_timesteps > 0:
    print(f"training from timestep {start_timesteps} / {TOTAL_TIMESTEPS}...")
    model.learn(
        total_timesteps=remaining_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    model.save(MODEL_PATH)
    print("model saved.")
else:
    print("model already trained for the total timesteps, skipping training.")

train_env.close()

# ---------------------
# Évaluation
# ---------------------
scores = []
total_rewards = []

for ep in range(10):
    eval_env = FlappyEnv(render_mode="human")
    obs, _ = eval_env.reset()
    done = False
    ep_reward = 0
    ep_score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        ep_reward += reward
        ep_score = info.get("score", ep_score)
        eval_env.render()

    eval_env.close()
    scores.append(ep_score)
    total_rewards.append(ep_reward)
    print(f"Episode {ep+1}: score={ep_score}, reward={ep_reward:.2f}")

print(f"\nAverage over 10 episodes: score={np.mean(scores):.2f}, reward={np.mean(total_rewards):.2f}")
