import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import flappy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# -----------------------------
# ENTORNO FLAPPY BIRD
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

        # Parámetros que se pueden ajustar con curriculum learning
        self.PIPE_GAP = 200
        self.PIPE_SPEED = 1.0

        # Inicializar superficie solo desde aquí
        #if self.render_mode == "human":
        #    if flappy.surface is None:  # Si aún no existe
        #        flappy.surface = pygame.display.set_mode((flappy.WIDTH, flappy.HEIGHT))

        #if render_mode == "human":
        #    flappy.surface = pygame.display.set_mode((flappy.WIDTH, flappy.HEIGHT))
        #else:
        #    flappy.surface = pygame.Surface((flappy.WIDTH, flappy.HEIGHT))

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
            img = np.clip(img, 0, 255).astype(np.uint8)
            pygame.surfarray.blit_array(pygame.display.get_surface(), img)
            pygame.display.flip()
            pygame.time.delay(30)

# -----------------------------
# CALLBACK CURRICULUM
# -----------------------------
#class CurriculumCallback(BaseCallback):
#    def __init__(self, verbose=1):
#        super().__init__(verbose)
#        self.max_speed = 7.0  # velocidad máxima de las tuberías

#    def _on_step(self) -> bool:
#        # Accedemos al entorno crudo dentro del Monitor y DummyVecEnv
#        env = self.training_env.envs[0].env  # Monitor -> FlappyEnv

        # Reducir PIPE_GAP progresivamente cada 5000 pasos
#        if self.num_timesteps % 5000 == 0:
#            env.PIPE_GAP = max(120, env.PIPE_GAP - 10)

        # Aumentar PIPE_SPEED progresivamente
#        progress = min(1.0, self.num_timesteps / 50_000)
#        env.PIPE_SPEED = min(self.max_speed, env.PIPE_SPEED + progress * (self.max_speed - env.PIPE_SPEED))

#        return True

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.max_speed = 7.0
        self.best_reward = -np.inf
        self.best_env_params = {}

    def _on_step(self) -> bool:
        # Accedemos al entorno original desempaquetando los wrappers
        env = self.training_env.envs[0].env
        while hasattr(env, "env"):
            env = env.env  # desempaqueta DummyVecEnv, Monitor, etc.

        # Reducir PIPE_GAP progresivamente cada 5000 pasos
        if self.num_timesteps % 5000 == 0:
            env.PIPE_GAP = max(120, env.PIPE_GAP - 10)

        # Aumentar PIPE_SPEED progresivamente
        progress = min(1.0, self.num_timesteps / 50_000)
        env.PIPE_SPEED = min(self.max_speed, env.PIPE_SPEED + progress * (self.max_speed - env.PIPE_SPEED))

        # Mostrar estado actual
        if self.num_timesteps % 5000 == 0:
            print(f"[Curriculum] TIMESTEPS: {self.num_timesteps}, PIPE_GAP: {env.PIPE_GAP}, PIPE_SPEED: {env.PIPE_SPEED}")

        # Guardar mejor recompensa
        if self.locals.get("rewards") is not None:
            ep_reward = np.sum(self.locals["rewards"])
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                self.best_env_params = {"PIPE_GAP": env.PIPE_GAP, "PIPE_SPEED": env.PIPE_SPEED}

        return True

    def _on_training_end(self) -> None:
        print("\n=== Mejor resultado alcanzado durante el entrenamiento ===")
        print(f"Recompensa: {self.best_reward}")
        print(f"Parámetros del entorno: {self.best_env_params}")




# -----------------------------
# ENTRENAMIENTO PPO CON CURRICULUM
# -----------------------------
if __name__ == "__main__":
    TIMESTEPS = 1_000_000  # puedes ajustar según tu tiempo de prueba

    # Creamos entorno con render para visualizar
    env = FlappyEnv(render_mode="human")
    env = Monitor(env)  # <-- Añadido: Monitor para calcular métricas por episodio
    env = DummyVecEnv([lambda: env])  # necesario para SB3

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=256, gamma=0.99)

    curriculum_callback = CurriculumCallback()

    model.learn(total_timesteps=TIMESTEPS, callback=curriculum_callback)

    # Guardamos modelo
    model.save("ppo_flappy_curriculum")

    env.close()
