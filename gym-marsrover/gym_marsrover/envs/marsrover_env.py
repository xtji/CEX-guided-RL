import os
import gym
import numpy
import random
import matplotlib.pyplot as plt
from operator import add
from matplotlib import colors
from gym.utils import seeding
from gym import error, spaces, utils
from matplotlib.image import imread


class MarsRover(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, image='marsrover_1.png'):
        self.background = imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), image))
        self.labels = self.background.__array__()
        self.initial_state = [35, 25]
        self.s = self.initial_state
        # range for the sine of action angle direction
        self.discrete_actions = [
            "right",
            "up",
            "left",
            "down"
        ]
        self.action_space = spaces.Discrete(4)
        self.continuous_action = [1, -1]
        self.nrow, self.ncol = self.background.shape[0], self.background.shape[1]
        self.rewards = {"goal": 1, "unsafe": 0, "safe": 0, "goal2": 0}
        self.obs_shape = numpy.array(self.initial_state).shape
        self.observation_space = spaces.Box(low=numpy.array([0, 0]), high=numpy.array([self.ncol, self.nrow]), shape=self.obs_shape)
        self.slip_probability = 0
        self.trace = []

    def reset(self):
        self.s = self.initial_state.copy()
        self.trace = []
        return self.s

    def step(self, action):
        if random.random() < self.slip_probability:
            action = self.action_space.sample()

        action = self.discrete_actions[action]
        traversed_distance = 10 * random.random()

        noise = [random.uniform(-0.1, 0.5), random.uniform(-0.1, 0.5), 0]
        if action == 'right':
            next_state = list(map(add, self.s, [0, traversed_distance]))
        elif action == 'up':
            next_state = list(map(add, self.s, [-traversed_distance, 0]))
        elif action == 'left':
            next_state = list(map(add, self.s, [0, -traversed_distance]))
        elif action == 'down':
            next_state = list(map(add, self.s, [traversed_distance, 0]))

        next_state = list(map(add, next_state, noise))

        # check for boundary violations
        if next_state[0] > self.nrow - 1:
            next_state[0] = self.nrow - 1
        if next_state[1] > self.ncol - 1:
            next_state[1] = self.ncol - 1
        if next_state[0] < 0:
            next_state[0] = 0
        if next_state[1] < 0:
            next_state[1] = 0

        next_state = [round(s, 2) for s in next_state]
        label = self.state_label(next_state)
        reward = self.get_reward(next_state)

        self.s = next_state
        self.trace.append(self.s)

        if label == 'goal' or label == 'unsafe':
            done = True
        else:
            done = False
        return next_state, reward, done, {}

    def state_label(self, state):
        # note: labels are inevitably discrete when reading an image file
        # thus in the following we look where the continuous state lies within
        # the image rgb matrix
        low_bound = [abs(state[i] - int(state[i])) for i in range(len(state))]
        high_bound = [1 - low_bound[i] for i in range(len(state))]
        state_rgb_indx = []
        for i in range(len(state)):
            if low_bound[i] <= high_bound[i]:
                # check for boundary
                if int(state[i]) > self.background.shape[i] - 1:
                    state_rgb_indx.append(self.background.shape[i] - 1)
                else:
                    state_rgb_indx.append(int(state[i]))
            else:
                # check for boundary
                if int(state[i]) + 1 > self.background.shape[i] - 1:
                    state_rgb_indx.append(self.background.shape[i] - 1)
                else:
                    state_rgb_indx.append(int(state[i]) + 1)

        if list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[0, 199]) or \
                list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[144, 199]):
            return 'unsafe'
        elif list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[45, 152]):
            return 'goal'
        elif list(self.labels[state_rgb_indx[0], state_rgb_indx[1]]) == list(self.labels[62, 16]):
            return 'goal2'
        else:
            return 'safe'

    def get_reward(self, state):
        label = self.state_label(state)
        return self.rewards[label]

    def render(self, mode='human'):
        pass