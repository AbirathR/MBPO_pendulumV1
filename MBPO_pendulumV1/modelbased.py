import time
import torch
import copy
import numpy as np
from utils import ReplayBuffer
from utils import Critic
from utils import Actor
from utils import Model


class ModelAgent:
    """
    Model-based agent for policy optimization.

    This agent uses a model of the environment to generate synthetic experiences
    and optimize the policy using these experiences.
    """
    def __init__(self, obs_dim, action_dim, *args, **kwargs):
        """
        Initializes the ModelAgent.

        Parameters:
        obs_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        """
        # Initialize arguments
        hidden_dims_actor = tuple(kwargs.get("hidden_dims_actor", (256, 256)))
        hidden_dims_critic = tuple(kwargs.get("hidden_dims_critic", (256, 256)))
        hidden_dims_model = tuple(kwargs.get("hidden_dims_model", (256, 256)))

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005
        self.delay = 2
        lr_actor = 0.001
        lr_critic = 0.001
        lr_model = 0.0001
        self.step_random = 500  # How many random actions to take before using actor for action selection
        self.update_every_n_steps = 51  # How often to update model, actor, and critics
        self.update_steps = 200  # How many gradient updates to perform per model when updating
        self.time = time.time()

        # Initialize actor
        self.actor = Actor(obs_dim, hidden_dims_actor, action_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_target.requires_grad_(False)

        # Initialize 2 critics
        self.critics = []
        self.critics_target = []
        self.optimizer_critics = []
        for _ in range(2):
            critic = Critic(obs_dim + action_dim, hidden_dims_critic)
            self.critics.append(critic)
            critic_target = copy.deepcopy(critic)
            critic_target.requires_grad_(False)
            self.critics_target.append(critic_target)
            self.optimizer_critics.append(torch.optim.Adam(critic.parameters(), lr=lr_critic))

        # Initialize models
        self.models = []
        self.optimizer_models = []
        for _ in range(25):
            model = Model(obs_dim + action_dim, hidden_dims_model, obs_dim)
            self.models.append(model)
            self.optimizer_models.append(torch.optim.Adam(model.parameters(), lr=lr_model))

        # Setup Replay Buffer
        self.buffer = ReplayBuffer()
        self.o_old = None
        self.a_old = None

        self.step_i = 0

    def reset(self):
        self.o_old = None
        self.a_old = None

    def loss_critic(self, val, target):
        """
        Computes the mean squared error loss for the critic network.
    
        Parameters:
        val (torch.Tensor): The predicted Q-values from the critic network.
        target (torch.Tensor): The target Q-values.
    
        Returns:
        torch.Tensor: The mean squared error loss.
        """
        diffs = target - val
        return torch.mean(diffs ** 2)

    def step(self, o, r, eval=False, done=False):
        """
        Takes a step in the environment, updates the replay buffer, and performs updates on the model, actor, and critics.
    
        Parameters:
        o (array-like): The current observation.
        r (float): The reward received from the environment.
        eval (bool): Flag indicating whether the step is for evaluation (default is False).
        done (bool): Flag indicating whether the episode is done (default is False).
        """
        o = torch.tensor(o, dtype=torch.float32)
        r = torch.tensor(float(r), dtype=torch.float32)
        done = torch.tensor(float(done), dtype=torch.float32)
        if not eval:
            # Add to replay buffer
            if self.o_old is not None:
                self.buffer.add((self.o_old, self.a_old, r, o, done))
    
            # Check if it's time to update the model, actor, and critics
            if self.step_i % self.update_every_n_steps == 0 and len(self.buffer) > self.buffer.n_samples:

                print("Performing update steps...")
                for _ in range(self.update_steps):
                    # Train Model
                    self.update_models()

                    # Train actor and critics using one minibatch of samples from every model
                    for model in self.models:
                        b = self.sample_from_model(model)

                        # Update Critic
                        self.update_critics(b)

                        # Update Actor
                        if self.step_i % self.delay == 0:
                            self.update_actor(b)

                            # Update Target Networks
                            self.update_target_networks()

            # Select Action
            with torch.no_grad():
                action = self.actor(o.unsqueeze(0)).squeeze()
                action_noisy = action + torch.randn_like(action) * 0.3
                action = torch.clamp(action_noisy, 0.0, 1.0)
            
            # Use random actions for the initial steps
            if self.step_i < self.step_random:
                action = torch.rand_like(action_noisy)
            self.o_old = o
            self.a_old = action if action.dim() > 0 else action.unsqueeze(0)
            self.step_i += 1
        else:
            # Select the action using the actor network without noise for evaluation
            with torch.no_grad():
                action = self.actor(o.unsqueeze(0)).squeeze()
                action = torch.clamp(action, 0.0, 1.0)

        return action.cpu().numpy()

    def update_target_networks(self):
        """
        Soft updates the target networks for the actor and critics.
    
        This method updates the target networks by blending the parameters of the
        target networks with the parameters of the main networks using a factor tau.
    
        The update rule is:
        target_param = tau * main_param + (1 - tau) * target_param
        """
        with torch.no_grad():
            # Update the target actor network
            for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                param_target.data.lerp_(param.data, self.tau)
            
            # Update the target critic networks
            for k in range(2):
                for param, param_target in zip(self.critics[k].parameters(), self.critics_target[k].parameters()):
                    param_target.data.lerp_(param.data, self.tau)

    def update_actor(self, b):
        """
        Updates the actor network using a batch of experiences.
    
        Parameters:
        b (dict): A dictionary containing batches of observations, actions, rewards, next observations, and done flags.
        """
        self.optimizer_actor.zero_grad()
        
        # Compute the Q-values for the current observations and actions predicted by the actor
        q_values = self.critics[0](b["o"], self.actor(b["o"]))
        
        # Compute the actor loss as the negative mean of the Q-values (Gradient Ascent)
        loss_actor = -torch.mean(q_values)
        loss_actor.backward()
        self.optimizer_actor.step()

    def update_critics(self, b):
        """
        Updates the critic networks using a batch of experiences.
    
        Parameters:
        b (dict): A dictionary containing batches of observations, actions, rewards, next observations, and done flags.
        """
        with torch.no_grad():
            # Compute the target actions using the target actor network
            a_target = self.actor_target(b["o_next"])
            
            # Add noise to the target actions for exploration
            noise = torch.clamp(torch.randn_like(a_target) * 0.1, -0.5, 0.5)
            a_target = torch.clamp(a_target + noise, 0.0, 1.0)
            
            # Compute the target Q-values using the target critic networks
            target_q_values = torch.min(
                *[critic_target(b["o_next"], a_target) for critic_target in self.critics_target]
            )
            
            # Compute the target values for the Q-function
            y = b["r"].unsqueeze(-1) + (1 - b["done"].unsqueeze(-1)) * self.gamma * target_q_values
    
        # Update each critic network
        for optimizer, critic in zip(self.optimizer_critics, self.critics):
            optimizer.zero_grad()
            q_values = critic(b["o"], b["a"])  # Compute the Q-values for the current state-action pairs
            loss = self.loss_critic(q_values, y)  # Compute the loss between the Q-values and the target values
            loss.backward()
            optimizer.step()

    def sample_from_model(self, model):
        """
        Samples a batch of experiences from the model.
    
        Parameters:
        model (Model): The model to sample experiences from.
    
        Returns:
        dict: A dictionary containing batches of observations, actions, next observations, and rewards.
        """
        # Sample a minibatch of experiences from the replay buffer
        batchsize = 128
        b = self.buffer.sample_tensors(n=batchsize)
  
        with torch.no_grad():
            action = self.actor(b["o"])
            
            # Add noise to the actions for exploration
            action_noisy = action + torch.randn_like(action) * 0.3
            b["a"] = torch.clamp(action_noisy, 0.0, 1.0)
        
        # Sample the next observations and rewards from the model using the generated actions
        new_o, r = model.sample(b["o"], b["a"])
        b["o_next"] = new_o
        b["r"] = r.squeeze()
        return b

    def update_models(self):
        samples = self.buffer.sample_tensors()
        for optim, model in zip(self.optimizer_models, self.models):
            self.model_step(model, optim, samples)

    def model_step(self, model, optim, samples):
        """
        Performs one gradient update for a model.
    
        Parameters:
        model (Model): The model to be updated.
        optim (torch.optim.Optimizer): The optimizer for the model.
        samples (dict): A dictionary containing batches of observations, actions, next observations, and rewards.
        """
        # Predict the next observation and reward using the model
        o_next_pred, r_pred = model(samples["o"], samples["a"])
        mu_o, sigma_o = o_next_pred
        mu_r, sigma_r = r_pred

        target_o = samples["o_next"]
        target_r = samples["r"].unsqueeze(1)
        
        # Compute the loss for the next observation prediction
        loss_o = torch.mean(((mu_o - target_o) ** 2) / (sigma_o ** 2) + torch.log(sigma_o ** 2))
        # Compute the loss for the reward prediction
        loss_r = torch.mean(((mu_r - target_r) ** 2) / (sigma_r ** 2) + torch.log(sigma_r ** 2))
        # Total loss is the sum of the observation and reward prediction losses
        loss = loss_o + loss_r
        # Perform a gradient update
        optim.zero_grad()
        loss.backward()
        optim.step()
