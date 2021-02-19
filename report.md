# Project 3: Report

### Implementation

My solution is implemented as a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) model with experience replay, following the methodology discussed in [Multi Agent Actor Critic for Mixed Cooperative Competitive environments](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf). The `Agent` class defines agents that use local observations of the environment's state to choose actions and learn from their experience. The `Actor` class defines the deep neural network for the policy, mapping observations of the local environment to continuous agent actions. The `Critic` class defines the deep neural network for the Deep Q-Learning Critics, mapping joint observations of the environment states and joint actions to action-values (Q-values). The `ActionNoise` class defines the process used to randomize actions at each step. The `ReplayBuffer` class stores past experiences that the agent samples from to update the model parameters. The `play_tennis()` function trains the agents and outputs scores for each episode.

### Learning Algorithm

The purpose of the learning algorithm is to parameterize a series of Actor and Critic models in the Tennis environment. Since there are two agents in the environment, each with local observations, I train two Actors and two Critics. The Actor models estimate the policy function for each agent using only local observations as inputs. The Critic models estimate the Q-value function using the joint observation and joint action space as inputs. After initializing these four networks the agents interact with the environment over a series of episodes. In each episode each agent considers its observation of the environment, takes actions and receives rewards. Using this data over many episodes the agents update their estimates of the action-value for each possible action and use the action-value estimates to improve their policies. This multi-Critic setup helps stabilize training by conditioning on all agent actions thereby ensuring the environment dynamics are stationary.

In my algorithm the agents interact with the environment by taking epsilon-greedy actions. Initially the agents act randomly selecting four random actions in the range [-1,1]. The `epsilon` value decays geometrically at a rate of 0.15% per episode until reaching a floor of 5%. Each action updates the state of the environment and returns some rewards. Then each experience for both agents is stored in the replay buffer. Every 2 steps through the environment the agents sample 64 experiences from the replay buffer. I experimented with a number of different epsilon decay rates, floors, batch sizes and update frequencies. Higher epsilon decay rates resulted in agents that weren't able to learn from the environment. Since the episodes contain relatively few experiences, the agents need to experiment over many episodes in order to learn actions. Lower floors on epsilon resulted in agents that stopped learning before solving the environment. Each of these hyper-parameters is closely related to the others, studying the relationships among them is an interesting topic for research.

Using these experiences the algorithm trains the four deep neural networks. The Critic models train the model parameters using the Adam optimizer with a learning rate equal to `0.0005`. The Critics' objective is to minimize the mean-squared error between the action-values predicted by the network (Q_expected) and an estimate of the true action-values (Q_targets). The true action-values (Q_targets) are estimated as the current reward plus the value of future expected rewards, discounted by `gamma = 0.99`. The Actor models train the model parameters using the Adam optimizer with a learning rate equal to `0.0003`. The Actors' objective is to maximize the mean action-values predicted by the Critic. Both the Actors and Critics use local and target networks to reduce the impact of correlation among sequential experiences. During each update step the agent updates the parameters of both target networks using the local networks with `tau = 0.001`. I experimented with a number of alternative learning rates and found that higher values for the learning rates destabilized the training.

The Critic models each take a state of size 48 and action of size 4 as input. While the actions are already in the range [-1,1], the states are range is [-30,30]. In each network I divide the state inputs by 9 to lower the standard deviation to approximately 1. The first layer of the Critic networks contains 128 nodes with ReLU activation and the second layer contains 64 nodes with ReLU activation. Finally each Critic network outputs the Q-value using ReLU activation. The Actor models each take an observation of size 24 as input. The first layer contains 128 nodes with ReLU activation and the second layer contains 64 nodes with ReLU activation. Finally each Actor network outputs two action values using tanh activation. I experimented with adding more nodes or fewer notes, but this architecture provide large enough to learn the behavior while being small enough to train in a reasonable period of time.

### Model summary

* BUFFER_SIZE = int(1e6)
* BATCH_SIZE = 64
* UPDATE_EVERY = 2
* GAMMA = 0.99
* TAU = 1e-3
* LR_ACTOR = 3e-4
* LR_CRITIC = 5e-4
* EPS = 1.0
* eps_decay = 0.9985
* eps_floor = 0.05
* actor_layers = (128, 64, 2)
* critic_layers = (128, 64, 2)`

### Plot of Rewards

After 4175 episodes, the successful agents were able to achieve an average score of +0.5 over its last 100 episodes.

![Scores](score_history.png)

### Ideas for Future Work

This project presents a number of potential avenues for future work. While my agents were able to solve the environment after nearly 5,000 episodes, very little improvement occurs between episodes 2000 and 3000 after the epsilon floor value was reached. This result suggests that I might be able to tune the hyper-parameters to speed learning. The agents may be spending too much time randomly exploring rather than learning how to improve the policy. Additionally I found that smaller networks struggle learn, so a deeper network might further improve learning. Mathematically, measuring the distance between two objects is a non-linear operation so larger networks may be better at learning these non-linearities in physical 2-D environments.

Multi-agent environments are both challenging and fascinating. The applications for multi-agent autonomous systems are quite vast and I found a number of interesting applications while doing research for this project. One example [Multi-Agent Deep Reinforcement Learning for Large-scale Traffic Signal Control](https://arxiv.org/pdf/1903.04527.pdf). The paradigm of centralized training with decentralized execution is a remarkable way to stabilize training and create dynamic agents. It is a topic I will continue to research after this class.
