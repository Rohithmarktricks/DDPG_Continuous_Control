import numpy as np
import random
import copy
from collections import namedtuple, deque

from new_model import Actor, Critic

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process


def get_device():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	return device

class Agent():

	def __init__(self, state_size, action_size, random_seed):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(random_seed)
		self.epsilon = EPSILON
		self.device = get_device()

		# Actor Network
		self.actor_local = Actor(state_size, action_size, random_seed).to(self.device)
		self.actor_target = Actor(state_size, action_size, random_seed).to(self.device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

		# Critic Network
		self.critic_local = Critic(state_size, action_size, random_seed).to(self.device)
		self.critic_target = Critic(state_size, action_size, random_seed).to(self.device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

		# noise for exploration
		self.noise = OUNoise(action_size, random_seed)

		# ReplayBuffer
		self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)


	def step(self, state, action, reward, next_state, done, timestep):

		self.memory.add(state, action, reward, next_state, done)

		if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
			for _ in range(LEARN_NUM):
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)


	def act(self, state, add_noise=True):
		state = torch.from_numpy(state).float().to(self.device)
		self.actor_local.eval()
		with torch.no_grad():
			actions = self.actor_local(state).cpu().data.numpy()

		self.actor_local.train()

		if add_noise:
			actions += self.epsilon * self.noise.sample()

		return np.clip(actions, -1, 1)

	def reset(self):
		self.noise.reset()


	def learn(self, experiences, gamma):
		# Q_targets = rewards + gamma * critic_targt(next_state, actor_target(next_state))
		
		states, actions, rewards, next_states, dones = experiences
		
		actions_next = self.actor_target(next_states)
		q_targets_next = self.critic_target(next_states, actions_next)
		
		q_targets = rewards + (gamma * q_targets_next * (1 - dones))

		# critic_loss
		q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(q_expected, q_targets)

		# minimize
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
		self.critic_optimizer.step()

		# actor_loss
		actor_preds = self.actor_local(states)
		actor_loss = self.critic_local(states, actor_preds).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()


		# update target networks
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)


		self.epsilon -= EPSILON_DECAY
		self.noise.reset()


	def soft_update(self, local_network, target_network, tau):
		for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class OUNoise():

	def __init__(self, size, seed, mu=0, theta=OU_THETA, sigma=OU_SIGMA):

		self.mu = mu*np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.reset()


	def reset(self):
		self.state = copy.copy(self.mu)


	def sample(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
		self.state = x+dx
		return self.state


class ReplayBuffer:

	def __init__(self, action_size, buffer_size, batch_size, seed):
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.experience = namedtuple("Experience", field_names = ['state', 'action', 'reward', 'next_state', 'done'])
		self.batch_size = batch_size
		self.seed = random.seed(seed)
		self.device = get_device()


	def add(self, state, action, reward, next_state, done):
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)


	def sample(self):

		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		return len(self.memory)




