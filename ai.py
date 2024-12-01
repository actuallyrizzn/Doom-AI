# AI for Doom

# Importing necessary libraries
import numpy as np  # For numerical computations and array manipulations
import torch  # PyTorch library for building and training neural networks
import torch.nn as nn  # PyTorch module for defining neural network layers
import torch.nn.functional as F  # Provides functions for activation and loss functions
import torch.optim as optim  # Optimizers for training neural networks
from torch.autograd import Variable  # Handles automatic differentiation for backpropagation

# Gym and Doom-specific libraries
import gym  # OpenAI Gym for creating RL environments
from gym.wrappers import SkipWrapper  # Wrapper to skip frames for faster training
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete  # Converts Doom's action space to discrete actions

# Importing helper modules from other files
import experience_replay  # Experience replay logic for storing and sampling transitions
import image_preprocessing  # Handles preprocessing of image inputs for the neural network

########## Building the AI ##########

# Defining the Convolutional Neural Network (CNN)
class CNN(nn.Module):  # Inherits from PyTorch's nn.Module base class
    """
    This class defines the "brain" of the AI using a Convolutional Neural Network (CNN).
    The network processes image inputs and outputs Q-values for each possible action.
    """

    def __init__(self, number_actions):
        """
        Initialize the CNN with convolutional layers and fully connected layers.
        :param number_actions: Number of actions the AI can choose from.
        """
        super(CNN, self).__init__()  # Call the constructor of the parent class (nn.Module)

        # Define convolutional layers
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)  # First convolutional layer
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  # Second convolutional layer
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)  # Third convolutional layer

        # Calculate the number of neurons after convolutions to set the input size of the fully connected layer
        flattened_size = self.count_neurons(image_dim=(1, 80, 80))  # For input images of size 80x80
        self.fc1 = nn.Linear(in_features=flattened_size, out_features=40)  # Fully connected hidden layer
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)  # Output layer for Q-values

    def count_neurons(self, *image_dim):
        """
        Dynamically calculate the number of neurons in the flattened layer after convolutions.
        This ensures the network can adapt to different image sizes.
        :param image_dim: Dimensions of the input image (channels, height, width).
        :return: Number of neurons in the flattened vector.
        """
        kernal_size = 3  # Kernel size for max pooling
        stride = 2  # Stride size for max pooling
        x = Variable(torch.rand(1, *image_dim))  # Simulate a random input image to determine output size

        # Apply convolutions and pooling to the simulated image
        x = F.relu(F.max_pool2d(self.convolution1(x), kernal_size, stride))  # First convolution + max pooling
        x = F.relu(F.max_pool2d(self.convolution2(x), kernal_size, stride))  # Second convolution + max pooling
        x = F.relu(F.max_pool2d(self.convolution3(x), kernal_size, stride))  # Third convolution + max pooling

        # Flatten the output to count the number of neurons
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        """
        Define the forward pass of the CNN. Takes an input image, processes it through the network,
        and outputs Q-values for each action.
        :param x: Input image.
        :return: Q-values for all possible actions.
        """
        kernal_size = 3  # Kernel size for max pooling
        stride = 2  # Stride size for max pooling
        x = F.relu(F.max_pool2d(self.convolution1(x), kernal_size, stride))  # First convolution + max pooling
        x = F.relu(F.max_pool2d(self.convolution2(x), kernal_size, stride))  # Second convolution + max pooling
        x = F.relu(F.max_pool2d(self.convolution3(x), kernal_size, stride))  # Third convolution + max pooling
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))  # Fully connected hidden layer
        x = self.fc2(x)  # Output layer
        return x


# Defining the "body" of the AI that selects actions based on Q-values
class SoftmaxAIBody(nn.Module):
    """
    This class defines the body of the AI, which interprets the CNN's outputs (Q-values)
    and selects actions probabilistically using the softmax function.
    """

    def __init__(self, temperature):
        """
        Initialize the body with a temperature parameter for softmax.
        :param temperature: Controls the randomness of action selection.
        """
        super(SoftmaxAIBody, self).__init__()
        self.T = temperature  # Temperature for softmax

    def forward(self, output_signals):
        """
        Select an action based on the output Q-values.
        :param output_signals: Q-values from the CNN.
        :return: Action chosen by the AI.
        """
        probabilities = F.softmax(output_signals * self.T)  # Compute probabilities using softmax
        actions = probabilities.multinomial()  # Sample actions based on probabilities
        return actions


# Combining the CNN and the body to form the complete AI
class AI:
    """
    This class combines the brain (CNN) and the body (SoftmaxAIBody) to create a complete AI agent.
    """

    def __init__(self, brain, body):
        """
        Initialize the AI with a brain and a body.
        :param brain: CNN that outputs Q-values.
        :param body: SoftmaxAIBody that selects actions based on Q-values.
        """
        self.brain = brain
        self.body = body

    def __call__(self, input_images):
        """
        Take input images, process them through the brain and body, and return actions.
        :param input_images: Preprocessed input images.
        :return: Actions selected by the AI.
        """
        input = Variable(torch.from_numpy(np.array(input_images, dtype=np.float32)))  # Convert images to PyTorch format
        output = self.brain(input)  # Get Q-values from the brain
        actions = self.body(output)  # Select actions using the body
        return actions.data.numpy()


########## Training the AI with Deep Convolutional Q-Learning ##########

# Setting up the Doom environment
doom_env = image_preprocessing.PreprocessImage(
    SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))),
    width=80,
    height=80,
    grayscale=True,
)  # Preprocess the Doom environment (skip frames, resize images, convert to grayscale)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force=True)  # Record gameplay for analysis
number_actions = doom_env.action_space.n  # Number of possible actions in the environment

# Building the AI
cnn = CNN(number_actions)  # Create the CNN
softmax_body = SoftmaxAIBody(temperature=1.0)  # Create the Softmax-based action selector
ai = AI(brain=cnn, body=softmax_body)  # Combine the brain and body to form the AI

# Setting up experience replay
n_steps = experience_replay.NStepProgress(env=doom_env, ai=ai, n_step=10)  # Progress over 10 steps
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)  # Replay memory with capacity 10,000

# Defining the eligibility trace for Q-learning updates
def eligibility_trace(batch):
    """
    Compute eligibility traces for n-step Q-learning.
    :param batch: Batch of transitions from experience replay.
    :return: Inputs and target Q-values for training.
    """
    gamma = 0.99  # Discount factor
    inputs = []  # Input states
    targets = []  # Target Q-values

    for series in batch:  # Iterate through each n-step series
        input = Variable(
            torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
        )
        output = cnn(input)  # Predict Q-values
        cumulative_reward = 0.0 if series[-1].done else output[1].data.max()  # Compute cumulative reward

        # Backtrack through the series to compute Q-value targets
        for step in reversed(series[:-1]):
            cumulative_reward = step.reward + gamma * cumulative_reward

        state = series[0].state  # Initial state
        target = output[0].data  # Predicted Q-values for the initial state
        target[series[0].action] = cumulative_reward  # Update the Q-value for the selected action
        inputs.append(state)  # Store the input state
        targets.append(target)  # Store the target Q-values

    return (
        torch.from_numpy(np.array(inputs, dtype=np.float32)),
        torch.stack(targets),
    )  # Return inputs and targets


# Moving average for tracking performance
class MovingAverage:
    """
    Tracks the moving average of rewards over a specified window size.
    """

    def __init__(self, size):
        """
        Initialize the moving average tracker.
        :param size: Window size for the moving average.
        """
        self.list_of_rewards = []  # List of rewards
        self.size = size  # Window size

    def add_reward_to_list(self, rewards):
        """
        Add new rewards to the list and maintain the window size.
        :param rewards: List or single reward to add.
        """
        if isinstance(rewards, list):  # Case 1: List of rewards
            self.list_of_rewards += rewards
        else:  # Case 2: Single reward
            self.list_of_rewards.append(rewards)

        # Remove old rewards if the list exceeds the window size
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average_rewards(self):
        """
        Compute the mean of the stored rewards.
        :return: Mean reward.
        """
        return np.mean(self.list_of_rewards)


# Initialize moving average tracker
ma = MovingAverage(100)

# Training the AI
loss = nn.MSELoss()  # Mean Squared Error loss function
optimizer = optim.Adam(cnn.parameters(), lr=0.001)  # Adam optimizer for training
nb_epochs = 100  # Number of training epochs

# Training loop
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)  # Collect 200 steps in memory
    for batch in memory.sample_batch(128):  # Train on batches of size 128
        inputs, targets = eligibility_trace(batch)  # Compute inputs and targets
        inputs, targets = Variable(inputs), Variable(targets)  # Convert to PyTorch variables
        predictions = cnn(inputs)  # Get predictions from the CNN
        loss_error = loss(predictions, targets)  # Compute loss
        optimizer.zero_grad()  # Reset gradients
        loss_error.backward()  # Backpropagate the loss
        optimizer.step()  # Update weights

    # Compute and track average rewards
    rewards_steps = n_steps.rewards_steps()
    ma.add_reward_to_list(rewards_steps)
    avg_reward = ma.average_rewards()
    print(f"Epoch: {epoch}, Average Reward: {avg_reward}")  # Print progress
