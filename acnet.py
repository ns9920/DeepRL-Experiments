import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy2(nn.Module):
    """
    implements a policy module for Actor-Critic
    """
    def __init__(self, ninputs, noutputs, gamma = 0.99):
        super(Policy2, self).__init__()
        self.fc1 = nn.Linear(ninputs, 128)
        
        self.actor = nn.Linear(128, noutputs)
        self.critic = nn.Linear(128, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

        # discount factor
        self.gamma = gamma
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        
        # smallest useful value
        self.eps = np.finfo(np.float32).eps.item()


    def forward(self, x):
        x = F.relu(self.fc1(x))
        # actor: choose action by returning probability of each action
        action_prob = F.softmax(self.actor(x), dim=1)
        # critic: evaluates being in the x
        state_values = self.critic(x)
        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state x 
        return action_prob, state_values
    
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()
        
    
    def update(self):
        R = 0
        returns = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # standardise returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        # compute policy losses, using stored log probs and returns
        loss = 0
        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss 
            policy_loss = -log_prob * advantage
            # calculate critic (value) loss using L1 smooth loss
            value_loss = F.smooth_l1_loss(value, torch.tensor([R]).unsqueeze(0))
            loss += (policy_loss + value_loss)
            
        # run backprop through all that
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # delete stored values 
        del self.rewards[:]
        del self.saved_actions[:]


ninputs = 3
noutputs = 4
policy = Policy2(ninputs, noutputs)

for episode in range(3):

    for step in range(100):
        state = np.random.normal(size=ninputs)
        action = policy.select_action(state)

        state = np.random.normal(size=ninputs)
        r = np.random.normal()    
        policy.rewards.append(r)
    
    policy.update()
    