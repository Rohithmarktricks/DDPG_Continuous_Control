'''
Source code for Actor and Critic networks.
@author: Rohith Banka.

This module contains the Network architecture for both Actor and Critic networks.
The Actor-Critic network architecture is taken from this paper:
https://arxiv.org/pdf/1509.02971.pdf

'''

# import modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    '''This function is invoked to fix the weights, for better learning'''
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128,
                 fc2_units=128):
        """Initialize parameters and build model.
        
        Params:
        =======
         state_size (int): Dimension of each state
         action_size (int): Dimension of each action
         seed (int): Random seed
         fc1_units (int): Number of nodes in first hidden layer
         fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # source: The low-dimensional networks had 2 hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        # applying a Batch Normalization on the first layer output
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc1))
        # source: The final layer weights and biases of the actor and were
        # initialized from a uniform distribution [−3 × 10−3, 3 × 10−3]
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        The total number of neurons in the output layer = total possible actions.
        since, the actor is DDPG Agent, the output is a real number (not probability, so no softmax layer)
        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        # source: used the rectified non-linearity for all hidden layers
        x = F.relu(self.fc1(state))
        # applying a batch Normalization on the first layer output
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # source The final output layer of the actor was a tanh layer,
        # to bound the actions
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128,
                 fc2_units=128):
        """Initialize parameters and build model.

        Params:
        ======
         state_size (int): Dimension of each state
         action_size (int): Dimension of each action
         seed (int): Random seed
         fcs1_units (int): Nb of nodes in the first hiddenlayer
         fc2_units (int): Nb of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        # applying a batch Normalization on the first layer output
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # source: The final layer weights and biases of the critic were
        # initialized from a uniform distribution [3 × 10−4, 3 × 10−4]
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values (just 1 output neuron)
         state: tuple.
         action: tuple.
        """
        xs = F.relu(self.fcs1(state))
        # applying a batch Normalization on the first layer output
        xs = self.bn1(xs)
        # source: Actions were not included until the 2nd hidden layer of Q
        x = torch.cat((xs, action.float()), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)