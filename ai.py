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
        layer by breaking the linearity with ReLU. Lastly, propagate the signals
        from the hidden layer to the output layer with the final output neurons.
        :param x: input image
        :return: output
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

# Making the body; the action based on the output from the brain (using softmax)
class SoftmaxAIBody(nn.Module):

    def __init__(self, temperature):
        super(SoftmaxAIBody, self).__init__()
        self.T = temperature

    def forward(self, output_signals):
        """
        Forward the output signal from the brain (the Q values contained in the output 
        neurons of the output layer) to the body of the AI which will play the action
        :param output_signals: The outputs from the brain
        :return: The action to be played
        """
        # use softmax to play the action
        probabilities = F.softmax(output_signals * self.T) # distribution of probabilities for each of our Q-values which depend on the input image and each action
        actions = probabilities.multinomial()
        return actions

# Assemble the brain and the body to make the AI 
class AI:
    
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
    
    def __call__(self, input_images):
        """
        Take the images as input and propagate the signals in the brain
        """
        # convert image into the right format
        input = Variable(torch.from_numpy(np.array(input_images, dtype = np.float32)))
        # take the brain and apply to input images
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()

########## Training the AI with Deep Convolutional Q-Learning ##########

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n 

# Building an AI

# Create the brain and body objects
cnn = CNN(number_actions)
softmax_body = SoftmaxAIBody(temperature=1.0)
ai = AI(brain=cnn, body=softmax_body)

# Setting up Experience Replay for learning every 10 steps
n_steps = experience_replay.NStepProgress(env=doom_env, ai=ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000) # Capacity, memory up to the last capacity amount of steps

# Implement eligibility trace
def eligibility_trace(batch):  # n-step q learning
    gamma = 0.99
    inputs = [] 
    targets = []
    for series in batch:  
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        # output = the prediction using the cnn
        output = cnn(input)
        cumulative_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumulative_reward = step.reward + gamma * cumulative_reward

        # Get inputs and targets ready
        state = series[0].state
        target = output[0].data # q value of the input state from the first transition
        target[series[0].action] = cumulative_reward
        input.append(state)
        targets.append(target)
    
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MovingAverage:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add_reward_to_list(self, rewards):
        """
        Add cumulative reward to list of rewards
        """
        # Case one: rewards is a list
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        # Case two: reward is a single element
        else:
            self.list_of_rewards.append(rewards)

        # Prevent list of rewards from exceeding 100
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0] 

    def average_rewards(self):
        """
        Return the mean of the list of rewards
        """
        return np.mean(self.list_of_rewards)

ma = MovingAverage(100)

# Training the AI
loss = nn.MSELoss() 
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 100
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128): # taking larger batch size because only sampling 10 steps at a time
        # Batch size is 128
        inputs, targets = eligibility_trace(batch)
        # Convert the inputs of the nn and the targets into torch variables
        inputs, targets = Variable(inputs), Variable(targets)
        # Get predictions
        predictions = cnn(inputs)
        # Get loss error
        loss_error = loss(predictions, targets)
        # Back propagate the loss error back into the nn
        optimizer.zero_grad() # Initialize
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add_reward_to_list(rewards_steps)
    avg_reward = ma.average_rewards()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
