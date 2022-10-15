import gymnasium as gym
import numpy as np
from itertools import count
import time

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

### main

seed = 1
gamma = 0.95
render = not False
finalrender = True
log_interval = 10
running_reward = 10

env = gym.make('CartPole-v1')
#env.seed(seed)
torch.manual_seed(seed)

ninputs = env.observation_space.shape[0]
noutputs = env.action_space.n
policy = Policy1(ninputs, noutputs)

starttime = time.time()

for i_episode in count(1):
    (state, _),  ep_reward = env.reset(), 0
    for t in range(1, 10000):  # Don't infinite loop while learning

        # select action from policy
        action = policy.select_action(state)

        # take the action
        state, reward, done, truncated, _ = env.step(action)
        reward = float(reward)     # strange things happen if reward is an int
        if render:
            env.render()
            
        policy.rewards.append(reward)
        ep_reward += reward
        if done or truncated:
            break

    # update cumulative reward
    running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

    # perform backprop
    policy.update()
    
    # log results
    if i_episode % log_interval == 0:
        print('Episode {}\t Last reward: {:.2f}\t Average reward: {:.2f}'.format(
            i_episode, ep_reward, running_reward))

    # check if we have solved cart pole
    if running_reward > env.spec.reward_threshold:
        secs = time.time() - starttime
        mins = int(secs/60)
        secs = round(secs - mins * 60.0, 1)
        print("Solved in {}min {}s!".format(mins, secs))
            
        print("Running reward is now {:.2f} and the last episode "
              "runs to {} time steps!".format(running_reward, t))

        if finalrender:
            state, _ = env.reset()
            for t in range(1, 10000):
                action = policy.select_action(state)
                state, reward, done, truncated, _ = env.step(action)
                env.render()
                if done or truncated:
                    break
        
        break
