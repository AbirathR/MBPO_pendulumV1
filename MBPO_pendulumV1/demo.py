import gymnasium as gym
from modelbased import ModelAgent
import random
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

def scale_action(env, action):
    low = env.action_space.low
    high = env.action_space.high
    scaled_action = low + (action + 1.0) * 0.5 * (high - low)
    return np.clip(scaled_action, low, high)

def run_episode(env, agent):
    obs, info = env.reset()
    reward_tot = 0.0
    done = False
    reward = 0.0
    while not done:
        action = scale_action(env, agent.step(obs, reward))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward_tot += reward
    agent.step(obs, reward)
    agent.reset()
    return reward_tot

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env = gym.make("Pendulum-v1", render_mode='rgb_array')
    env = RecordVideo(env, video_folder='logging/', episode_trigger=lambda episode_id: True)
    agent = ModelAgent(env.observation_space.shape[0], env.action_space.shape[0])

    for i in range(50):
        reward_tot = run_episode(env, agent)
        print("Episode:", i+1, "--- Total Reward:", reward_tot)
    env.close()
