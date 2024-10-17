import gymnasium as gym
from modelbased import ModelAgent
import random
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

import numpy as np

def scale_action(env, action):
    """
    Scales an action from the range [-1, 1] to the action space of the environment.

    Parameters:
    env (gym.Env): The environment with a defined action space.
    action (float or np.ndarray): The action to be scaled, expected to be in the range [-1, 1].

    Returns:
    np.ndarray: The scaled action, clipped to the action space of the environment.
    """
    low = env.action_space.low
    high = env.action_space.high
    
    # Scale the action from [-1, 1] to the [low, high] range of the action space
    scaled_action = low + (action + 1.0) * 0.5 * (high - low)
    
    # Clip the scaled action to ensure it is within the action space bounds
    return np.clip(scaled_action, low, high)

def run_episode(env, agent):
    """
    Runs a single episode in the gym environment using the policy agent.

    Parameters:
    env (gym.Env): The environment to run the episode in.
    agent (ModelAgent): The agent that interacts with the environment.

    Returns:
    float: The total reward accumulated during the episode.
    """
    obs, info = env.reset()
    reward_tot = 0.0
    done = False
    reward = 0.0
    
    # Loop until the episode is not terminated or truncated
    while not done:
        # Get the action from the agent and scale it to the environment's action space
        action = scale_action(env, agent.step(obs, reward))
        
        # Take a step in the environment using the scaled action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if the episode is terminated or truncated
        done = terminated or truncated
        
        # Accumulate the reward (no discounting for simplicity)
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
        # Run a single episode and get the total reward
        reward_tot = run_episode(env, agent)
        print("Episode:", i + 1, "--- Total Reward:", reward_tot)
    env.close()
