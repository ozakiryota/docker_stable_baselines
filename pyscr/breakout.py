import gymnasium as gym
import sb3_contrib
import os


env_id = 'ALE/Breakout-v5'
weights_path = '../save/Breakout_QRDQN'

if os.path.exists(weights_path + '.zip'):
    model = sb3_contrib.QRDQN.load(weights_path)
else:
    training_env = gym.make(env_id)
    model = sb3_contrib.QRDQN('CnnPolicy', training_env, verbose=1, buffer_size=int(1e+4))
    model.learn(total_timesteps=1e+6, progress_bar=True)
    model.save(weights_path)

evaluation_env = gym.make(env_id, render_mode='human')
obs, _ = evaluation_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True, progress_bar=True)
    obs, reward, is_terminated, is_truncated, info = evaluation_env.step(action)
    evaluation_env.render()
    if is_terminated or is_truncated:
        obs, _ = evaluation_env.reset()
evaluation_env.close()