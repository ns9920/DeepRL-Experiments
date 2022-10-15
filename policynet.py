import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy1(nn.Module):
    """
    implements a policy module for REINFORCE
    """
    def __init__(self, ninputs, noutputs, gamma = 0.99):
        super(Policy1, self).__init__()

        self.fc1 = nn.Linear(ninputs, 128)
        self.fc2 = nn.Linear(128, noutputs)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

        # discount factor
        self.gamma = gamma
        # action & reward buffer
        self.saved_log_probs = []
        self.rewards = []
        
        # smallest useful value
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        scores = self.fc2(x)
        return F.softmax(scores, dim=1)
    
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
        
    
    def update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # standardise returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        # compute policy losses, using stored log probs and returns
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # run backprop through all that
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # delete stored values 
        del self.rewards[:]
        del self.saved_log_probs[:]


ninputs = 3
noutputs = 4
policy = Policy1(ninputs, noutputs)

for episode in range(300):

    for step in range(100):
        state = np.random.normal(size=ninputs)
        action = policy.select_action(state)

        state = np.random.normal(size=ninputs)
        r = np.random.normal()    
        policy.rewards.append(r)
    
    policy.update()
    