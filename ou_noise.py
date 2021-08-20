"""Base class for OU Noise.
Noise is added to the actions (outputs of Actor Network) to facilitate exploration.
"""

import random
import copy
import numpy as np

# Hyperparameters of OUNOise BaseClass.
OU_SIGMA = 0.2  # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15  # Ornstein-Uhlenbeck noise parameter


class OUNoise():
	"""Source code for Ornstein-Uhlenbeck Process"""

	def __init__(self, size, seed, mu=0, theta=OU_THETA, sigma=OU_SIGMA):
		"""Initializes the input parameters and the noise process, this noise is added to the
		actions of the actor network, to facilitate the exploration process"""
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		random.seed(seed)
		self.state = 0
		self.reset()

	def reset(self):
		"""Resets the internal state of the noise to Mean, mu"""
		self.state = copy.copy(self.mu)

	def sample(self):
		"""Update the internal state of the noise and return the sample"""
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
		self.state = x + dx
		return self.state
