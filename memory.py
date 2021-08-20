'''
The input to DDPG agent is vector based observation. We store experience tuple 
in this Replay Buffer as the agent is interacting with the environment and then 
sample a small batch of tuples from it in order to learn.

As, a result, the agent will be able to learn from individual tuple multiple times,
recall rare occurrences/interactions, and in general make better use of past experiences.

@author : Rohith Banka

Project for Udacity NanoDegree in Deep Reinforcement Learning (DRLND)

code expanded and adaptyed from code examples provided by Udacity DRL Team, 2021.

Other References: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

'''


import numpy as np
from collections import namedtuple, deque
from utils import get_device
import torch

class ReplayBuffer:

	def __init__(self, action_size, buffer_size, batch_size, seed):
		'''Replay Buffer class to store the agent experiences in the environment.'''
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.experience = namedtuple("Experience", field_names = ['state', 'action', 'reward', 'next_state', 'done'])
		self.batch_size = batch_size
		self.seed = random.seed(seed)
		self.device = get_device()


	def add(self, state, action, reward, next_state, done):
		'''Adds state, action, reward, next_state and done status to the memory pool.'''
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)


	def sample(self):
		'''This method randomly samples a batch of experiences from memory, generate tensors and move them to the same device, where Agent is available'''
		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		return len(self.memory)