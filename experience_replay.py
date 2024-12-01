# Experience Replay

# Importing necessary libraries
import numpy as np  # For numerical computations and array manipulations
from collections import namedtuple, deque  # For structured data storage and efficient queue operations

# Defining a single step in the environment
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])
"""
This namedtuple defines the structure of a single transition in the environment:
- `state`: The environment's state before the action.
- `action`: The action taken by the agent.
- `reward`: The reward received after the action.
- `done`: Whether the episode terminated after this step.
"""

# Implementing n-step progress tracking
class NStepProgress:
    """
    This class tracks the AI's progress over multiple steps (n-step) in the environment.
    It yields sequences of transitions for training purposes.
    """

    def __init__(self, env, ai, n_step):
        """
        Initialize the n-step tracker.
        :param env: The environment instance.
        :param ai: The AI agent that selects actions.
        :param n_step: The number of steps to track in each sequence.
        """
        self.ai = ai  # The AI agent
        self.rewards = []  # To store cumulative rewards for completed episodes
        self.env = env  # The environment
        self.n_step = n_step  # Number of steps in each sequence

    def __iter__(self):
        """
        Create an iterator that yields n-step transitions from the environment.
        :return: Tuples of `Step` objects representing n-step transitions.
        """
        state = self.env.reset()  # Initialize the environment and get the initial state
        history = deque()  # A queue to store the history of steps
        reward = 0.0  # Accumulate rewards for an episode

        while True:  # Loop until the environment stops
            action = self.ai(np.array([state]))[0][0]  # AI selects an action based on the current state
            next_state, r, is_done, _ = self.env.step(action)  # Take the action in the environment
            reward += r  # Accumulate reward
            history.append(Step(state=state, action=action, reward=r, done=is_done))  # Add step to history

            # Ensure the history doesn't exceed n_step + 1
            while len(history) > self.n_step + 1:
                history.popleft()  # Remove the oldest step

            # If history length reaches n_step + 1, yield the oldest n steps
            if len(history) == self.n_step + 1:
                yield tuple(history)

            state = next_state  # Move to the next state

            # Handle end-of-episode scenarios
            if is_done:
                # If the history is still larger than n_step, remove the oldest step
                if len(history) > self.n_step + 1:
                    history.popleft()

                # Yield remaining transitions in history
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()

                # Store the total reward for this episode
                self.rewards.append(reward)
                reward = 0.0  # Reset reward for the next episode
                state = self.env.reset()  # Reset the environment for the next episode
                history.clear()  # Clear the history

    def rewards_steps(self):
        """
        Retrieve and clear the list of cumulative rewards for episodes.
        :return: A list of cumulative rewards.
        """
        rewards_steps = self.rewards  # Store current rewards
        self.rewards = []  # Reset rewards list
        return rewards_steps


# Implementing Experience Replay for Reinforcement Learning
class ReplayMemory:
    """
    This class manages experience replay, storing transitions and providing random batches for training.
    """

    def __init__(self, n_steps, capacity=10000):
        """
        Initialize the replay memory.
        :param n_steps: An iterator of n-step transitions.
        :param capacity: Maximum number of transitions to store in memory.
        """
        self.capacity = capacity  # Maximum number of transitions to store
        self.n_steps = n_steps  # The n-step iterator
        self.n_steps_iter = iter(n_steps)  # Create an iterator from the n-steps
        self.buffer = deque()  # A queue to store transitions

    def sample_batch(self, batch_size):
        """
        Create an iterator that returns random batches of transitions from the memory.
        :param batch_size: Number of transitions in each batch.
        :return: Randomly shuffled batches of transitions.
        """
        ofs = 0  # Offset to track batches
        vals = list(self.buffer)  # Convert buffer to a list for shuffling
        np.random.shuffle(vals)  # Randomize the order of transitions

        # Yield batches of the specified size
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size : (ofs + 1) * batch_size]
            ofs += 1  # Move to the next batch

    def run_steps(self, samples):
        """
        Add transitions to the replay memory by sampling from the n-steps iterator.
        :param samples: Number of transitions to add to memory.
        """
        while samples > 0:  # Keep adding until the desired number of samples is reached
            entry = next(self.n_steps_iter)  # Get the next n-step transition
            self.buffer.append(entry)  # Add the transition to the buffer
            samples -= 1  # Decrement the sample counter

        # If the buffer exceeds its capacity, remove the oldest transitions
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()  # Remove the oldest transition
