import flappy as flappy
import numpy as np
import pygame
from stable_baselines3 import PPO

# -----------------------------
# Environnement minimal Flappy
# -----------------------------
class FlappyEnv:
    def __init__(self):
        pygame.init()
        if flappy.surface is None:
            flappy.surface = pygame.display.set_mode((flappy.WIDTH, flappy.HEIGHT))
            pygame.display.set_caption("Flappy Bird RL")

    def reset(self):
        flappy.rl_init()
        return np.array(flappy.rl_obs(), dtype=np.float32), {"score": flappy.score}

    def step(self, action):
        reward, done = flappy.rl_step(action)
        obs = np.array(flappy.rl_obs(), dtype=np.float32)
        info = {"score": flappy.score}
        return obs, reward, done, False, info

    def render(self):
        # Rendu graphique
        img = flappy.rl_render()
        img = np.clip(img, 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(pygame.display.get_surface(), img)
        pygame.display.flip()
        pygame.time.delay(0)


# -----------------------------
# Évaluation
# -----------------------------
if __name__ == "__main__":
    model = PPO.load("ppo_flappy_curriculum")
    env = FlappyEnv()
    num_episodes = 10

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            env.render()  # <- ici on voit le jeu
        print(f"Episode {ep+1}: score={info['score']}, reward={total_reward}")

    pygame.quit()
