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
        """ 
        CNN with 3 convolutional layers and 1 hidden layer 
        """
        
        super(CNN, self).__init__() # Activate inheritance to use tools from nn.Module
        # Convolution connections
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5) # applies convolution to the input images
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3) 
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2) 
        # Flatten pixels obtained by the convolutions that were applied to get a vector
        # Vector that will be used as the input for the NN
        self.fc1 = nn.Linear(in_features=self.count_neurons(image_dim=(1, 80, 80)), out_features=40) # Full connection between input layer (vector) and hidden layer
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions) # Full connection between the hidden layer and output layer composed on the output neurons that correspond to a Q value of the possible actions
        
    def count_neurons(self, *image_dim):
        """ 
        Count the number of pixels, which represent the number of neurons
        in the vectpr after the convolutions are applied 
        """
        kernal_size = 3
        stride = 2
        x = Variable(torch.rand(1, *image_dim)) # Input image
        # Propagate image into the NN to reach the flattening layer to get the number of neurons we want
        x = F.relu(F.max_pool2d(self.convolution1(x), kernal_size, stride)) # First convolutional layer with max pooling and ReLU activation
        x = F.relu(F.max_pool2d(self.convolution2(x), kernal_size, stride)) # Propagate images from first to second convolutional layer
        x = F.relu(F.max_pool2d(self.convolution3(x), kernal_size, stride)) # Propagate images from second to third convolutional layer
        # Flatten pixels of third convolutional layer
        # Flattening layer
        return x.data.view(1, -1).size(1) # Takes all pixels of all the third layer channels and puts them in a vector (input of the fully connected network)

    def forward(self, x):
        """
        Propage the signals from the flattening layer to the hidden layer
        of the fully connected network. Then activate the neurons of this hidden 
        layer by breaking the linearity with ReLU. :astly, propagate the signals
        from the hidden layer to the output layer with the final output neurons.
        @param x: input image
        """
        kernal_size = 3
        stride = 2
        x = F.relu(F.max_pool2d(self.convolution1(x), kernal_size, stride)) # First convolutional layer with max pooling and ReLU activation
        x = F.relu(F.max_pool2d(self.convolution2(x), kernal_size, stride)) # Propagate images from first to second convolutional layer
        x = F.relu(F.max_pool2d(self.convolution3(x), kernal_size, stride)) # Propagate images from second to third convolutional layer
        x = x.view(x.size(0), -1) # Flattening layer
        x = F.relu(self.fc1(x)) # Pass signal with linear transmission and break linearity with rectifier function ReLU
        x = self.fc2(x)
        return x

# Making the body



# Assemble the brain and the body to make the AI 




########## Training the AI with Deep Convolutional Q-Learning ##########

