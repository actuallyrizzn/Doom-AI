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
        # Convolution connections
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5) # applies convolution to the input images
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) 
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2) 
        # Flatten pixels obtained by the convolutions that were applied to get a vector
        # Vector that will be used as the input for the NN
        self.fc1 = nn.Linear(in_features=self.count_neurons(), out_features=40) # Full connection between input layer (vector) and hidden layer
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions) # Full connection between the hidden layer and output layer composed on the output neurons that correspond to a Q value of the possible actions
        
    def count_neurons(self):
        """ Count the number of pixels """
        return 0

# Making the body



# Assemble the brain and the body to make the AI 




########## Training the AI with Deep Convolutional Q-Learning ##########

