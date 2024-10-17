import torch
import random
from torch import nn

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.n_samples = 128
        self.max_size = 1_000_000

    def __len__(self):
        return len(self.buffer)

    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) > self.max_size:
            del self.buffer[0]

    def sample(self, n=None):
        n = n or self.n_samples
        samples = random.choices(self.buffer, k=n)
        data = list(zip(*samples))
        data_dict = {
            "o": data[0],
            "a": data[1],
            "r": data[2],
            "o_next": data[3],
            "done": data[4]
        }
        return data_dict

    def sample_tensors(self, n=None):
        n = n or self.n_samples
        samples = random.choices(self.buffer, k=n)
        data = list(zip(*samples))
        data_dict = {
            "o": torch.stack(data[0]),
            "a": torch.stack(data[1]),
            "r": torch.stack(data[2]).squeeze(),
            "o_next": torch.stack(data[3]),
            "done": torch.stack(data[4]).squeeze()
        }
        return data_dict

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Tanh())  # Action space is between -1 and 1
        self.net = nn.Sequential(*layers)

    def forward(self, observation):
        return self.net(observation)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        return self.net(x)

class Model(nn.Module):
    """
    Contains a probabilistic world model. Outputs two tuples:
    - (mu_o, sigma_o): mean and standard deviation of the next observation
    - (mu_r, sigma_r): mean and standard deviation of the reward
    """
    def __init__(self, input_dim, hidden_dims, obs_dim):
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
        x = torch.cat([observation, action], dim=-1)
        h = self.net(x)
        mu_o = self.mu_output(h)
        sigma_o = torch.exp(self.log_sigma_output(h))
        mu_r = self.mu_reward(h) * 5  # Scaling factor as in original code
        sigma_r = torch.exp(self.log_sigma_reward(h))
        return (mu_o, sigma_o), (mu_r, sigma_r)

    def sample(self, observation, action):
        with torch.no_grad():
            (mu_o, sigma_o), (mu_r, sigma_r) = self.forward(observation, action)
            new_o = mu_o + torch.randn_like(mu_o) * sigma_o
            r = mu_r + torch.randn_like(mu_r) * sigma_r
        return new_o, r
