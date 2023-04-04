# AI for Doom

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing



########## Building the AI ##########

# Making the brain
class CNN(nn.Module):

    def __init__(self, number_actions):
        """ CNN with 3 convolutional layers and 1 hidden layer """
        
        super(CNN, self).__init__() # Activate inheritance to use tools from nn.Module
        self.convolution1 # Convolution connections
        self.convolution2
        self.convolution3
        # Flatten pixels obtained by the convolutions that were applied to get a vector
        # Vector that will be used as the input for the NN
        self.fc1
        self.fc2
        


# Making the body



# Assemble the brain and the body to make the AI 




########## Training the AI with Deep Convolutional Q-Learning ##########

