
# Solution to Project 3: Collaboration and Competition

### Introduction

This repository contains my solution for the third project in Udacity's Deep Reinforcement Learning Nanodegree. For this project I trained two agents to solve the Tennis environment by controlling their respective rackets. In this environment a reward of +0.1 is provided if the agent hits the ball over the net and a reward of -0.01 is provided if the ball drops to the ground or is hit out of bounds. The goal of my agents is to keep the ball in play by cooperating to hit it back-and-forth over the net.

<p align="center"> <img src="tennis.png" width="400"> </p>

The observation space includes 24 variables corresponding to the position and velocity of the ball and the rackets. Each agent has its own local observation space so the joint observation space has 48 variables total. The actions available to each agent are to either move left and right [-1,1] or up and down [-1,1], represented as a vector of length two for each agent. The environment is considered solved once the agents achieve an average score of +0.5 over 100 consecutive episodes.

### Getting Started

My solution was coded using Python version 3.6.12, PyTorch version 0.4.0 and OpenAI gym version 0.17.3.

1. The requirements for running my solution are available in the Udacity [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in `README.md` at the root of the repository.

2. Additionally you will need to download the [Tennis environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip). My solution was created on a Windows (64-bit) machine and trained using a GPU.

3. After downloading the file you may copy this repository, place the Reacher environment file in the project directory and unzip it. 

### Instructions

After completing the initial setup, the Jupyter Notebook `Tennis_Solution.ipynb` contains my solution. The notebook references three supporting python files. The `udacity_agent.py` file contains the definition of the agents, the `udacity_model.py` file contains the structure of the deep neural network model and the `udacity_buffer.py` file contains the replay buffer. Finally the estimated model parameters that solved the environment are located in the `actor0_local.pth`, `actor1_local.pth`, `critic0_local.pth` and `critic1_local.pth` files. The `report.md` file contains a descrption of the algorithm used to estimate these parameters.
