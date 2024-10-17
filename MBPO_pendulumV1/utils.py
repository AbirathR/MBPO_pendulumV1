import torch
import random
from torch import nn

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experiences.
    """

    def __init__(self):
        """
        Initializes the replay buffer.
        """
        self.buffer = []
        self.n_samples = 128
        self.max_size = 1_000_000

    def __len__(self):
        """
        Returns the current size of the buffer.
        """
        return len(self.buffer)

    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.max_size:
            del self.buffer[0]

    def sample(self, n=None):
        """
        Samples a batch of experiences from the buffer.

        Parameters:
        n (int): Number of samples to return. Defaults to self.n_samples.

        Returns:
        dict: A dictionary containing batches of observations, actions, rewards, next observations, and done flags.
        """
        n = n or self.n_samples
        samples = random.choices(self.buffer, k=n)
        data = list(zip(*samples))
        data_dict = {
            "o": data[0],  # Observations
            "a": data[1],  # Actions
            "r": data[2],  # Rewards
            "o_next": data[3],  # Next observations
            "done": data[4]  # Done flags
        }
        return data_dict

    def sample_tensors(self, n=None):
        """
        Samples a batch of experiences from the buffer and converts them to tensors.

        Parameters:
        n (int): Number of samples to return. Defaults to self.n_samples.

        Returns:
        dict: A dictionary containing batches of observations, actions, rewards, next observations, and done flags as tensors.
        """
        n = n or self.n_samples
        samples = random.choices(self.buffer, k=n)
        data = list(zip(*samples))
        data_dict = {
            "o": torch.stack(data[0]),  # Observations as tensors
            "a": torch.stack(data[1]),  # Actions as tensors
            "r": torch.stack(data[2]).squeeze(),  # Rewards as tensors
            "o_next": torch.stack(data[3]),  # Next observations as tensors
            "done": torch.stack(data[4]).squeeze()  # Done flags as tensors
        }
        return data_dict

class Actor(nn.Module):
    """
    Actor network for policy approximation.

    This network takes an observation as input and outputs an action.
    The action space is assumed to be between -1 and 1.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        Initializes the Actor network.

        Parameters:
        input_dim (int): Dimension of the input (observation space).
        hidden_dims (tuple): Dimensions of the hidden layers.
        output_dim (int): Dimension of the output (action space).
        """
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Tanh())  # Action space is between -1 and 1
        self.net = nn.Sequential(*layers)

    def forward(self, observation):
        """
        Forward pass through the network.

        Parameters:
        observation (torch.Tensor): The input observation.

        Returns:
        torch.Tensor: The output action.
        """
        return self.net(observation)

class Critic(nn.Module):
    """
    Critic network for value function approximation.

    This network takes an observation and action as input and outputs a Q-value.
    """
    def __init__(self, input_dim, hidden_dims):
        """
        Initializes the Critic network.

        Parameters:
        input_dim (int): Dimension of the input (observation + action space).
        hidden_dims (tuple): Dimensions of the hidden layers.
        """
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, observation, action):
        """
        Forward pass through the network.

        Parameters:
        observation (torch.Tensor): The input observation.
        action (torch.Tensor): The input action.

        Returns:
        torch.Tensor: The output Q-value.
        """
        x = torch.cat([observation, action], dim=-1)
        return self.net(x)

class Model(nn.Module):
    """
    Contains a probabilistic world model. Outputs two tuples:
    - (mu_o, sigma_o): mean and standard deviation of the next observation
    - (mu_r, sigma_r): mean and standard deviation of the reward
    """
    def __init__(self, input_dim, hidden_dims, obs_dim):
        """
        Initializes the Model network.

        Parameters:
        input_dim (int): Dimension of the input (observation + action space).
        hidden_dims (tuple): Dimensions of the hidden layers.
        obs_dim (int): Dimension of the observation space.
        """
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        self.net = nn.Sequential(*layers)

        self.mu_output = nn.Linear(dims[-1], obs_dim)
        self.log_sigma_output = nn.Linear(dims[-1], obs_dim)
        self.mu_reward = nn.Linear(dims[-1], 1)
        self.log_sigma_reward = nn.Linear(dims[-1], 1)

    def forward(self, observation, action):
        """
        Forward pass through the network.

        Parameters:
        observation (torch.Tensor): The input observation.
        action (torch.Tensor): The input action.

        Returns:
        tuple: Two tuples containing the mean and standard deviation of the next observation and reward.
        """
        x = torch.cat([observation, action], dim=-1)
        h = self.net(x)
        
        # Compute the mean and standard deviation of the next observation
        mu_o = self.mu_output(h)
        sigma_o = torch.exp(self.log_sigma_output(h))
        
        # Compute the mean and standard deviation of the reward
        mu_r = self.mu_reward(h) * 5  # Scaling factor as in original code
        sigma_r = torch.exp(self.log_sigma_reward(h))
        return (mu_o, sigma_o), (mu_r, sigma_r)

    def sample(self, observation, action):
        """
        Samples the next observation and reward from the model.

        Parameters:
        observation (torch.Tensor): The input observation.
        action (torch.Tensor): The input action.

        Returns:
        tuple: The sampled next observation and reward.
        """
        with torch.no_grad():
            (mu_o, sigma_o), (mu_r, sigma_r) = self.forward(observation, action)
            new_o = mu_o + torch.randn_like(mu_o) * sigma_o
            r = mu_r + torch.randn_like(mu_r) * sigma_r
        return new_o, r
