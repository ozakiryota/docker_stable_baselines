import gymnasium as gym
import sb3_contrib
import os


env_id = 'Humanoid-v4'
weights_path = '../save/Humanoid_TQC'

if os.path.exists(weights_path + '.zip'):
    model = sb3_contrib.TQC.load(weights_path)
else:
    training_env = gym.make(env_id)
    model = sb3_contrib.TQC('MlpPolicy', training_env, verbose=1)
    model.learn(total_timesteps=1e+6, progress_bar=True)
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