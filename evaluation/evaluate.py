import time
import numpy as np
from stable_baselines3 import PPO
from flappyEnv.flappy_env import FlappyEnv

def evaluate(model_path="model/ppo_flappy_curriculum", num_episodes=5, sleep_time=0.02):

    print(f"[INFO] loading model : {model_path}")
    model = PPO.load(model_path)
    print("[INFO] loaded.")

    env = FlappyEnv(render_mode="human")
    print("[INFO] env initalized (render=human).")

    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0

        print(f"\n---- Episode {ep} ----")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward

            env.render()
            time.sleep(sleep_time)

        print(f"→ total Reward episode {ep} = {ep_reward}")
        episode_rewards.append(ep_reward)

    env.close()

    print("\n=== final result ===")
    print(f"Mean: {np.mean(episode_rewards):.2f} | Std: {np.std(episode_rewards):.2f}")

    return episode_rewards


if __name__ == "__main__":
    evaluate(
        num_episodes=10
    )
