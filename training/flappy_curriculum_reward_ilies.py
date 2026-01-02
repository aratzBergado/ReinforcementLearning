import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os

import flappyEnv.flappy_for_curriculum as flappy

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------
# variable globale pour la vitesse de départ
START_SPEED = 1.0

# -----------------------------
# ENV GYM
# -----------------------------
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
            pygame.init()
            if flappy.surface is None:
                flappy.surface = pygame.display.set_mode(
                    (flappy.WIDTH, flappy.HEIGHT)
                )

    def reset(self, seed=None, options=None):
        global START_SPEED
        flappy.rl_init(
            pipe_speed=START_SPEED,
            pipe_gap_min=flappy.game_state.get("PIPE_GAP_MIN", 250),
            pipe_gap_max=flappy.game_state.get("PIPE_GAP_MAX", 350),
        )
        print(f"Début d'un nouvel épisode | pipe_speed = {START_SPEED}")
        return np.array(flappy.rl_obs(), dtype=np.float32), {}

    def step(self, action):
        total_reward = 0
        done = False

        for _ in range(self.frame_skip):
            r, done = flappy.rl_step(action)
            total_reward += r
            if done:
                break

        return (
            np.array(flappy.rl_obs(), dtype=np.float32),
            total_reward,
            done,
            False,
            {"score": flappy.game_state["score"]},
        )

    def render(self):
        if self.render_mode == "human":
            img = flappy.rl_render()
            pygame.surfarray.blit_array(pygame.display.get_surface(), img)
            pygame.display.flip()


# -----------------------------
# CURRICULUM CALLBACK
# -----------------------------
class RewardCurriculumCallback(BaseCallback):
    def __init__(self, min_gap=130, max_speed=7.0, threshold_reward=150):
        super().__init__()
        self.min_gap = min_gap
        self.max_speed = max_speed
        self.threshold_reward = threshold_reward
        self.reward_growth = 5
        self.curriculum_speed = 1.0  # vitesse persistante

    def _on_step(self) -> bool:
        global START_SPEED
        infos = self.locals.get("infos")
        if infos is None:
            return True

        for info in infos:
            if "episode" not in info:
                continue

            ep_reward = info["episode"]["r"]

            if ep_reward >= self.threshold_reward:
                # augmentation du gap
                flappy.game_state["PIPE_GAP_MIN"] = max(self.min_gap, flappy.game_state["PIPE_GAP_MIN"] - 5)
                flappy.game_state["PIPE_GAP_MAX"] = max(self.min_gap, flappy.game_state["PIPE_GAP_MAX"] - 5)

                # augmentation de la vitesse de départ
                self.curriculum_speed = min(self.max_speed, self.curriculum_speed + 0.1)
                flappy.game_state["speed"] = self.curriculum_speed
                START_SPEED = self.curriculum_speed  # mise à jour globale pour le prochain reset

                print(f"\nLEVEL UP | reward={ep_reward} "
                      f"gap=[{flappy.game_state['PIPE_GAP_MIN']}, {flappy.game_state['PIPE_GAP_MAX']}] "
                      f"speed={flappy.game_state['speed']}")

                self.threshold_reward += self.reward_growth

        env.render()
        return True


# -----------------------------
# TRAINING
# -----------------------------
if __name__ == "__main__":
    TIMESTEPS = 300_000

    env = FlappyEnv(render_mode=None)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]

    if checkpoints:
        latest_checkpoint = max(
            checkpoints,
            key=lambda x: int(x.split("_")[-2])
        )
        print(f"latest checkpoint loading : {latest_checkpoint}")
        model = PPO.load(
            os.path.join(checkpoint_dir, latest_checkpoint),
            env=env
        )
    else:
        print("No checkpoint found. Starting from scratch")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=256,
            gamma=0.99,
        )

    curriculum_callback = RewardCurriculumCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix="flappy_curriculum",
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
    )

    model.save("ppo_flappy_curriculum")
    env.close()