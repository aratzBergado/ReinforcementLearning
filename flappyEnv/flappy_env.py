import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import flappyEnv.flappy as flappy

 
class FlappyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, frame_skip=4):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, 0], dtype=np.float32),
            high=np.array([flappy.HEIGHT, 20, flappy.WIDTH, flappy.HEIGHT], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(2)
        if render_mode == "human":
            flappy.surface = pygame.display.set_mode((flappy.WIDTH, flappy.HEIGHT))
        else:
            flappy.surface = pygame.Surface((flappy.WIDTH, flappy.HEIGHT))

    def reset(self, seed=None, options=None):
        flappy.rl_init()
        obs = np.array(flappy.rl_obs(), dtype=np.float32)
        info = {"score": flappy.score}
        return obs, info

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            reward, done = flappy.rl_step(action)
            total_reward += reward
            if done:
                break
        obs = np.array(flappy.rl_obs(), dtype=np.float32)
        info = {"score": flappy.score}
        return obs, total_reward, done, False, info


    def render(self):
        if self.render_mode == "human":
            img = flappy.rl_render()
            pygame.surfarray.blit_array(pygame.display.get_surface(), img)
            pygame.display.flip()
