import os
from stable_baselines3 import PPO
from flappyEnv.flappy_env import FlappyEnv
import numpy as np

MODEL_PATH = "ppo_flappy"

train_env = FlappyEnv(render_mode=None, frame_skip=3)

if os.path.exists(MODEL_PATH + ".zip"):
    print("existing model loading...")
    print("training...")
    model = PPO.load(MODEL_PATH, env=train_env)
    model.learn(total_timesteps=50000)
    model.save(MODEL_PATH)
    print("model saved.")
else:
    print("no model existing, creating new...")
    print("training...")
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save(MODEL_PATH)
    print("model saved.")

train_env.close()

scores = []
total_rewards = []

for ep in range(10):
    eval_env = FlappyEnv(render_mode="human", frame_skip=3)
    obs, _ = eval_env.reset()
    done = False
    ep_reward = 0
    ep_score = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = eval_env.step(action)
        ep_reward += reward
        ep_score = info.get("score", ep_score)
        eval_env.render()

    eval_env.close()
    scores.append(ep_score)
    total_rewards.append(ep_reward)
    print(f"Episode {ep+1}: score={ep_score}, reward={ep_reward:.2f}")

print(f"\nAverage over 10 episodes: score={np.mean(scores):.2f}, reward={np.mean(total_rewards):.2f}")