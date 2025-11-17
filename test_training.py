import os
from stable_baselines3 import PPO
from flappy_env import FlappyEnv

MODEL_PATH = "ppo_flappy"

train_env = FlappyEnv(render_mode=None, frame_skip=3)

if os.path.exists(MODEL_PATH + ".zip"):
    print("existing model loading...")
    model = PPO.load(MODEL_PATH, env=train_env)
else:
    print("no model existing, creating new...")
    print("training...")
    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(MODEL_PATH)
    print("model saved.")

train_env.close()

eval_env = FlappyEnv(render_mode="human", frame_skip=3)
obs, _ = eval_env.reset()
done = False
total_reward = 0
score = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = eval_env.step(action)
    total_reward += reward
    score = info.get("score", score)
    eval_env.render()

eval_env.close()
print(f"evaluation done : final score = {score}, total reward = {total_reward:.2f}")