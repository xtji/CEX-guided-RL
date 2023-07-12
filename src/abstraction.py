import gym
import math
import itertools
import numpy
from gym.wrappers import TimeLimit
from collections import defaultdict

class Abstraction(gym.Wrapper):

    def __init__(self, env, cfg, counterexample_policy={}):
        super(Abstraction, self).__init__(env)
        self.policy = counterexample_policy
        self.MDP = TimeLimit(env, max_episode_steps=cfg.hyper.iteration_num)
        self.cfg = cfg
        self.s = self.MDP.reset()
        self.state_num = 0
        self.ap = ["unsafe", "pre-unsafe", "goal", "goal1", "safe"]
        self.labelling_function = {label: [] for label in self.ap}
        self.abstract_state = {}
        self.information = {}
        if hasattr(self.MDP, "ncol"):
            self.ncol, self.nrow = self.MDP.ncol, self.MDP.nrow
        else:
            self.ncol, self.nrow = 8, 8
        self.box_bounds = [[], []]
        self.Q = {}
        self.result = []
        self.history = {"s": [], "action": [], "s_a": [], "sprime": [], "label": []}
        self.positions = []
        self.rewards = []
        self.violates = []
        self.offlines = []
        self.n_envs = cfg.hyper.n_envs
        self.current_actions = [0 for i in range(self.n_envs)]
        self.simulation = False
        self.penalty = cfg.env.penalty
        self.episode = 0
        self.offline_episode = 0
        self.sim_episode_num = 0
        self.model = None
        self.concrete_transition_history = {}
        self.set_cover_result = None
        self.trace = []

    def reset(self):
        self.episode_reward = 0
        obs = self.MDP.reset()
        self.s = obs
        return obs

    def data_collection(self, state, action, next_state, reward, done, info):
        discretised_state = str([int(s) for s in state])
        self.concrete_transition_history.setdefault(discretised_state, {a: {} for a in range(self.MDP.action_space.n)})
        self.concrete_transition_history[discretised_state][action][str(next_state)] = \
            (self.concrete_transition_history[discretised_state][action].get(str(next_state), (0, None, None))[0] + 1,
             reward, done)

        label = 'safe'
        if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
            self.violates.append(1)
            self.rewards.append(0)
            self.episode += 1
            self.offlines.append(self.offline_episode)
        elif done:
            self.episode += 1
            if reward <= 0:
                self.violates.append(0)
                label = 'unsafe'
            else:
                self.violates.append(1)
                label = 'goal'

            self.rewards.append(reward)
            self.offlines.append(self.offline_episode)

        self.history["s"].append(state)
        self.history["action"].append(action)
        self.history["s_a"].append(state + [action])
        self.history["sprime"].append(next_state)
        self.history["label"].append(label)

    def prob_matrix(self, history, rounding_digits=2):
        action_space = range(self.MDP.action_space.n)

        abstract_sprime = [self.get_state(sprime) for sprime in history["sprime"]]
        history["abstract_sprime"] = abstract_sprime

        history["s"] = history["s"].apply(tuple)
        history["sprime"] = history["sprime"].apply(tuple)
        grouped_history = history.groupby("s")

        transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        labelling_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        possible_matrix = defaultdict(set)

        for s, group in grouped_history:
            abstract_s = self.get_state(s)
            state_num = int(abstract_s.strip('[').strip(']').split(",")[0])
            if all(abstract_sprime == abstract_s for abstract_sprime in group["abstract_sprime"]):
                continue

            action_info = group.groupby("action")

            for action, action_group in action_info:
                sprime_info = action_group.groupby("abstract_sprime")
                for sprime, count_info in sprime_info:
                    #reduce self-loop in transition matrix for simpler optimisation
                    if sprime == abstract_s:
                        count = 0.01
                    else:
                        count = len(count_info)
                    next_state_num = int(sprime.strip('[').strip(']').split(",")[0])
                    possible_matrix[state_num].add(next_state_num)
                    transition_matrix[abstract_s][action][sprime] += count
                    next_label = self.get_label(sprime)
                    labelling_matrix[state_num][action][next_label] += count

        terminal_state = set()
        for abstract_s in transition_matrix:
            for action in action_space:
                total = sum(transition_matrix[abstract_s][action].values())
                if total > 0:
                    for sprime in transition_matrix[abstract_s][action]:
                        if sprime not in transition_matrix:
                            terminal_state.add(sprime)
                        transition_matrix[abstract_s][action][sprime] /= total
                        transition_matrix[abstract_s][action][sprime] = round(
                            transition_matrix[abstract_s][action][sprime], rounding_digits)
                else:
                    transition_matrix[abstract_s][action][abstract_s] = 1

        for terminal_s in terminal_state:
            transition_matrix[terminal_s] = {a: {terminal_s: 1} for a in action_space}

        for abstract_s in labelling_matrix:
            for action in action_space:
                total = sum(labelling_matrix[abstract_s][action].values())
                if total > 0:
                    for next_label in labelling_matrix[abstract_s][action]:
                        labelling_matrix[abstract_s][action][next_label] /= total
                        labelling_matrix[abstract_s][action][next_label] = round(
                            labelling_matrix[abstract_s][action][next_label], rounding_digits)
                else:
                    for label in self.labelling_function:
                        if abstract_s in self.labelling_function[label]:
                            labelling_matrix[abstract_s][action][label] = 1

        self.transition_matrix = transition_matrix
        self.labelling_matrix = labelling_matrix
        self.possible_matrix = possible_matrix
        return transition_matrix

    def get_label(self, abstract_state):
        if isinstance(abstract_state, str):
            state = [int(s) for s in abstract_state.strip('[').strip(']').split(",")][0]
        else:
            state = int(abstract_state[0])
        for label, states in self.labelling_function.items():
            if state in states:
                return label
        return ValueError

    def get_state(self, state):
        if str(state).isdigit():
            state = [int(state) // self.MDP.ncol, int(state) % self.MDP.ncol]
        elif isinstance(state, str):
            state = [float(s) for s in state.strip('[').strip(']').split(",")]

        for l, states in self.labelling_function.items():
            for s in states:
                for bound in self.abstract_state[s]:
                    if hasattr(self.MDP, "flag") and all(bound[2 * i] <= state[i] < bound[2 * i + 1] for i in range(len(state) - 1)):
                        return str([s, int(state[-1])])
                    elif not hasattr(self.MDP, "flag") and all(bound[2 * i] <= state[i] < bound[2 * i + 1] for i in range(len(state))):
                        return str(s)
        return ValueError

    def get_position(self, obs):
        if str(obs).isdigit():
            position = [obs // self.ncol, obs % self.nrow]
        else:
            position = obs
        return position

    def experience_replay(self, state):
        trace = []
        done = False
        iteration = 0
        while not done:
            abstract_state = self.get_state(state)

            if abstract_state in self.policy:
                action = list(self.policy[abstract_state].keys())[0]
            else:
                trace = []
                break

            next_state, reward, done = self.predict_step(state, action)

            if done:
                reward = self.cfg.env.penalty

            iteration += 1

            if not next_state or iteration > self.cfg.hyper.iteration_num:
                done = True
                trace = []
            else:
                trace.append((next_state, reward, done))
                state = next_state
        if not trace:
            self.sim_episode_num += 5
        else:
            self.sim_episode_num += 1
        self.trace = trace
        return trace

    def predict_step(self, obs, action):
        obs = self.get_position(obs)
        abstract_state = self.get_state(obs)

        discretised_state = [int(s) for s in obs]

        if str(discretised_state) not in self.concrete_transition_history or not \
        self.concrete_transition_history[str(discretised_state)][action]:
            return None, None, None

        target_abstract_state = []
        possible_outputs = {}

        if abstract_state in self.policy and action in self.policy[abstract_state]:
            for t in self.policy[abstract_state][action]:
                target_abstract_state.append(t)
        target_abstract_state.append(abstract_state)

        for t in target_abstract_state:
            possible_outputs = {cs: dist for cs, dist in self.concrete_transition_history[str(discretised_state)][action].items() if self.get_state(self.get_position(cs)) == t}

            if possible_outputs:
                break

        if not possible_outputs:
            return None, None, None

        next_states = list(possible_outputs.keys())
        weights = list(v[0] / sum([v[0] for v in possible_outputs.values()]) for v in possible_outputs.values())
        next_state_str = numpy.random.choice(next_states, p=weights)
        next_state = [float(s) for s in next_state_str.strip('[').strip(']').split(",")]

        if self.cfg.env.state_type == "continuous":
            next_state = [round(next_state[i], self.cfg.hyper.rounding_digits) for i in range(len(next_state))]
        else:
            next_state = [round(next_state[i]) for i in range(len(next_state))]

        next_state[0] = round(next_state[0], 2)
        next_state[1] = round(next_state[1], 2)
        if next_state[0] > self.nrow - 1:
            next_state[0] = self.nrow - 1
        if next_state[1] > self.ncol - 1:
            next_state[1] = self.ncol - 1
        if next_state[0] < 0:
            next_state[0] = 0
        if next_state[1] < 0:
            next_state[1] = 0

        _, reward, done = possible_outputs[next_state_str]
        return next_state, reward, done

    def sim_step(self):
        obs, reward, done = self.trace[0]
        self.trace.pop(0)
        info = {}
        return obs, reward, done, info

    def step(self, action):
        if self.simulation:
            while not self.trace and self.sim_episode_num < self.cfg.hyper.sim_episode_num:
                self.experience_replay(self.s)
            if not self.trace:
                obs, reward, done, info = self.s, 0, True, {}
            else:
                obs, reward, done, info = self.sim_step()
        else:
            state = self.get_position(self.MDP.s)
            obs, reward, done, info = self.MDP.step(action)
            next_state = self.get_position(obs)

            self.data_collection(state, action, next_state, reward, done, info)

        self.s = obs
        return obs, reward, done, info

