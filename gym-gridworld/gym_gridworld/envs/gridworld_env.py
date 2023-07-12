import gym
import numpy
import random
from operator import add
from gym import spaces


class GridEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, mode):
        self.mode = mode
        self.nrow, self.ncol = 40, 40
        self.flag = 0
        self.actions = [
            "right",
            "up",
            "left",
            "down"
        ]
        self.action_space = spaces.Discrete(4)
        self.mu, self.sigma = 2, 0.5
        self.s = [0, 39, 0]
        self.slip_probability = 0.15
        self.bounds = [0, 8, 25, 33, 39], [0, 8, 12, 28, 39]
        self.rewards = {"goal": 1, "unsafe": 0, "safe": 0, "goal1": 0}
        self.obs_shape = numpy.array(self.s).shape
        self.observation_space = spaces.Box(low=numpy.array([0, 0, 0]), high=numpy.array([self.nrow, self.ncol, 1]), shape=self.obs_shape)

    def step(self, action):
        if random.random() < self.slip_probability:
            action = self.action_space.sample()

        if self.mode == 'discrete':
            noise = 1
        elif self.mode == 'hybrid':
            noise = abs(random.gauss(2, 0.5))
        action = self.actions[action]

        # grid movement dynamics:
        if action == 'right':
            next_state = list(map(add, self.s, [0, noise, 0]))
        elif action == 'up':
            next_state = list(map(add, self.s, [-noise, 0, 0]))
        elif action == 'left':
            next_state = list(map(add, self.s, [0, -noise, 0]))
        elif action == 'down':
            next_state = list(map(add, self.s, [noise, 0, 0]))

        if next_state[0] > self.nrow - 1:
            next_state[0] = self.nrow - 1
        if next_state[1] > self.ncol - 1:
            next_state[1] = self.ncol - 1
        if next_state[0] < 0:
            next_state[0] = 0
        if next_state[1] < 0:
            next_state[1] = 0

        # update current state
        next_state = [round(s, 2) for s in next_state]
        label = self.state_label(next_state)
        reward = self.get_reward(next_state)

        if label == 'goal1' and self.flag == 0:
            self.flag = 1
            next_state[-1] = self.flag

        self.s = next_state

        if (label == 'goal' and self.flag == 1) or label == 'unsafe':
            done = True
        else:
            done = False
        return next_state, reward, done, {}

    def reset(self):
        self.flag = 0
        self.s = [0, 39, 0]
        return self.s

    def state_label(self, state):
        if 33 <= state[0] <= 39 and 0 <= state[1] <= 39:
            return 'goal'
        elif 0 <= state[0] < 8 and 0 <= state[1] < 8:
            return 'goal1'
        elif 25 <= state[0] < 33 and 12 <= state[1] < 28:
            return 'safe'
        elif 25 <= state[0] < 33 and 0 <= state[1] <= 39:
            return 'unsafe'
        else:
            return 'safe'

    def get_reward(self, state):
        label = self.state_label(state)
        if (label == 'goal' and self.flag == 0) or (label == 'goal1' and self.flag == 1):
            label = 'safe'
        return self.rewards[label]

    def render(self, mode='human'):
        pass