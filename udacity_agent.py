
import numpy as np
import random

from udacity_model import Actor, Critic
from udacity_replay_buffer import ReplayBuffer, ActionNoise

import torch
import torch.nn.functional as F
import torch.optim as optim

# Modeling hyper-parameters
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
UPDATE_EVERY = 2
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 3e-4
LR_CRITIC = 5e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Function to convert data in lists to float tensors
def torchify(flatlist_data):
    return torch.from_numpy(np.vstack(flatlist_data)).float()

class Agent():

    def __init__(self, actor_size, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_size = actor_size
        self.seed = random.seed(random_seed)

        # Actor Networks have access to only local states
        self.actors_local = [Actor(state_size, action_size, random_seed).to(device) for _ in range(actor_size)]
        self.actors_target = [Actor(state_size, action_size, random_seed).to(device) for _ in range(actor_size)]
        self.actors_optimizer = [optim.Adam(self.actors_local[n].parameters(), lr=LR_ACTOR) for n in range(actor_size)]

        # Critic Networks have access to all states and actions
        self.critics_local = [Critic(state_size*actor_size, action_size*actor_size, random_seed).to(device) for _ in range(actor_size)]
        self.critics_target = [Critic(state_size*actor_size, action_size*actor_size, random_seed).to(device) for _ in range(actor_size)]
        self.critics_optimizer = [optim.Adam(self.critics_local[n].parameters(), lr=LR_CRITIC) for n in range(actor_size)]

        # Time step for when to update
        self.t_step = 0
        
        # Noise process - do all actors use the same noise during training?
        self.noise = ActionNoise(action_size, random_seed)

        # Replay memory will be indexed by actor
        self.memory = ReplayBuffer(actor_size, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    # Add experience for all agents to environment and learn
    def step(self, observations, actions, rewards, next_observations, dones):
        self.memory.add(observations, actions, rewards, next_observations, dones)

        if len(self.memory) > BATCH_SIZE and self.t_step % UPDATE_EVERY==0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
           
        self.t_step += 1

    # Use states to determine action from the actor
    def act(self, observations, eps, add_noise=True):
        actions = []
        
        # Loops through actors to generate actions
        for n in range(self.actor_size):
            observation = torch.from_numpy(observations[n]).float().to(device)
            self.actors_local[n].eval()
            with torch.no_grad():
                action = self.actors_local[n](observation).cpu().data.numpy()
            self.actors_local[n].train()
            
            # Select random action
            if eps > random.random() and add_noise:
                action = self.noise.sample()
            actions.append(action)
            
        return actions

    # Learn from experiences
    def learn(self, experiences, gamma):
        
        states, observations, actions, rewards, next_states, next_observations, dones = experiences
        
        states = torchify(states).to(device)
        actions = torchify(actions).to(device)
        rewards = torchify(rewards).to(device)
        next_states = torchify(next_states).to(device)
        dones = torchify(dones).to(device)
        
        # Loops through actors / critics to update parameters
        for n in range(self.actor_size):

            # Actors use their respective observations to generate predicted actions and next actions
            obs = [torchify([o[n] for o in observations]).to(device) for n in range(self.actor_size)]
            actions_pred = [self.actors_local[n](obs[n]) for n in range(self.actor_size)]
            actions_pred = torch.cat(actions_pred, 1)
        
            next_obs = [torchify([o[n] for o in next_observations]).to(device) for n in range(self.actor_size)]
            actions_next = [self.actors_target[n](next_obs[n]) for n in range(self.actor_size)]
            actions_next = torch.cat(actions_next, 1)
            
            # Generate rewards for each actor
            agent_rewards = torchify([r[n] for r in rewards]).to(device)
            
            # Get predicted next-state actions and Q values from target models
            Q_targets_next = self.critics_target[n](next_states, actions_next)
            Q_targets = agent_rewards + (gamma * Q_targets_next * (1 - dones))
            
            # Compute critic loss and minimize
            Q_expected = self.critics_local[n](states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            self.critics_optimizer[n].zero_grad()
            critic_loss.backward()
            self.critics_optimizer[n].step()
                    
            # Compute actor loss and minimize
            actor_loss = -self.critics_local[n](states, actions_pred).mean()
            self.actors_optimizer[n].zero_grad()
            actor_loss.backward()
            self.actors_optimizer[n].step()
    
            self.soft_update(self.critics_local[n], self.critics_target[n], TAU)
            self.soft_update(self.actors_local[n], self.actors_target[n], TAU)

    # Performs a soft update from local model to target model by factor TAU
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
