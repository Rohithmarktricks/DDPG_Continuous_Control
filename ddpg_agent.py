'''

DDPG Agent Source Code
@author: Rohith Banka

Continuous control done as a part of Deep Reinforcement Learning Nanodegree.
Initial Code has been taken from https://github.com/ShangtongZhang/DeepRL/blob/master/examples.py

This main source code for Agent, to interact with the environment.
The input or the state is a vector.


Actions:
=======
The simulation contains a single agent i.e., a double-jointed arm that can move to target locations.
At each time step, the agent can take an action (continuous). The action is a vector of 4 numbers, corresponding
to the torque applicable at 2 joints. Each entry in the action vector/tensor should be a number between -1 and 1.


States:
=======
The state space consists of 33 variables (a tensor of 33 entries) corresponding to position, rotation,
velocity, and angular velocities of the arm.


Learning:
========
DDPG with Actor-Critic scinario has been used to train the agent. 
Since DDPG is an off-policy algorithm, we will have local network (to explore the environment) and 
target network to learn the policy. So, we have Critic network (local and target) and Actor network( local and target)

'''

# import important modules
import numpy as np
import random
import copy
from collections import namedtuple, deque
from memory import ReplayBuffer
from ou_noise import OUNoise
from model import Actor, Critic
from utils import get_device
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Hyperparameters
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # Gamma; discount factor
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


class Agent():
	'''
	Agent class, to instantiate the agent that interacts with and learns from the environment.'''
	
	def __init__(self, state_size, action_size, random_seed):
		'''
		Initializes an agent object.

		Input Params:
		=============
		state_size(int): dimension of the state
		action_size(int): dimension of the action
		random_seed(int): random seed
		'''
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

		# step to have both local and target networks the same weights.
		self.hard_copy_weights(self.actor_target, self.actor_local)
		self.hard_copy_weights(self.critic_target, self.critic_local)


	def step(self, state, action, reward, next_state, done, timestep):
		'''Save experience namedtuple in replaybuffer and use random sample from buffer to learn.
		experience tuple has (state, action, reward, next_state, done)'''

		self.memory.add(state, action, reward, next_state, done)

		if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
			for _ in range(LEARN_NUM):
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)


	def act(self, state, add_noise=True):
		'''Returns the actions for given state as per current policy

		Params:
		======
		state(int): input to the network.
		add_noise(boolean): To add the noise.
		'''
		state = torch.from_numpy(state).float().to(self.device)
		# eval mode to get the actions from network and disable the gradient tracking.
		self.actor_local.eval()
		with torch.no_grad():
			actions = self.actor_local(state).cpu().data.numpy()

		# set the network back to train mode, it also enables the gradient tracking.
		self.actor_local.train()

		# The actions of the network are not stochastic, but deterministic and to 
		# increase the exploration probability, we add noise to the actions (outputs of the network).
		if add_noise:
			actions += self.epsilon * self.noise.sample()

		# we clip the final outputs to range(-1, 1)
		return np.clip(actions, -1, 1)

	def reset(self):
		'''To reset the noise distribution'''
		self.noise.reset()


	def learn(self, experiences, gamma):
		'''
		Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        '''
		
		
		states, actions, rewards, next_states, dones = experiences
		
		'''================ Update Critic Network ====================='''
		actions_next = self.actor_target(next_states)
		
		# Q value for next state,action pair
		q_targets_next = self.critic_target(next_states, actions_next)
		
		# TD value
		q_targets = rewards + (gamma * q_targets_next * (1 - dones))

		# critic_loss
		q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(q_expected, q_targets)

		# minimize the critic loss
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
		self.critic_optimizer.step()

		'''================= Update Actor Netwrork ====================='''
		# get the possible action values
		actor_preds = self.actor_local(states)
		# get the Qvalue of the (state, action)
		actor_loss = self.critic_local(states, actor_preds).mean()
		# minimize the actor loss.
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()


		# update target networks
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)

		# for Exploration and Exploitation
		self.epsilon -= EPSILON_DECAY
		self.noise.reset()


	def soft_update(self, local_network, target_network, tau):
		'''Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_network : PyTorch model (weights will be copied from)
            target_network : PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        '''
		for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


	def hard_copy_weights(self, target, source):
		'''This function is invoked as a part of the initialization of the target networks,
		and this step is to make sure that the target and local networks have same weights'''
		for target_param, local_param in zip(target.parameters(), source.parameters()):
			target_para.data.copy_(local_param.data)





