import gymnasium as gym
import stable_baselines3
import os


env_id = 'CartPole-v1'
weights_path = '../save/CartPole_DQN'

if os.path.exists(weights_path + '.zip'):
    model = stable_baselines3.DQN.load(weights_path)
else:
    training_env = gym.make(env_id)
    model = stable_baselines3.DQN('MlpPolicy', training_env, verbose=1)
    model.learn(total_timesteps=1e+5, progress_bar=True)
    model.save(weights_path)

evaluation_env = gym.make(env_id, render_mode='human')
obs, _ = evaluation_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_terminated, is_truncated, info = evaluation_env.step(action)
    evaluation_env.render()
    if is_terminated or is_truncated:
        obs, _ = evaluation_env.reset()
evaluation_env.close()