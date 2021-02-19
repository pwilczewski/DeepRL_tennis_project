
import numpy as np
import random
import copy
import itertools

import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Replay buffer stores experiences that are sampled from during learning
class ReplayBuffer:

    def __init__(self, actor_size, action_size, buffer_size, batch_size, seed):
        self.actor_size = actor_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "observations", "actions", "rewards", "next_states", "next_observations", "dones"])
        self.seed = random.seed(seed)

    # Add experience to memory
    def add(self, observations, actions, rewards, next_observations, dones):
        # Flatten lists of states and actions over actors, critics will use the joint state / action space
        states = list(itertools.chain(*observations))
        actions = list(itertools.chain(*actions))
        next_states = list(itertools.chain(*next_observations))
        dones = int(max(dones))
        
        e = self.experience(states, observations, actions, rewards, next_states, next_observations, dones)
        self.memory.append(e)

    # Sample experiences from memory
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        t_experiences = map(list, zip(*experiences))
        
        return t_experiences

    def __len__(self):
        return len(self.memory)

# Uniform random noise process seems to work best
class ActionNoise:
    
    def __init__(self, size, mu=0.):
        self.size = size
        self.mu = mu * np.ones(size)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        self.state = np.array([np.random.uniform(-1,1) for _ in range(self.size)])
        return self.state

