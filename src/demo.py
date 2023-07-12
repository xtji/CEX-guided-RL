import sys
import gym
import math
import pandas
import numpy as np
import warnings
import scipy.stats as stats
from tqdm import tqdm
from collections import defaultdict
from stable_baselines3.common.type_aliases import RolloutReturn

sys.path.append("..")
from src.branchandbound import *
from src.abstraction import Abstraction
from src.counterexample import minsetCounterexampleCVX

warnings.filterwarnings('ignore')

def branchandbound(MDP, uncovered_unsafe_points, safe_points, unsafe_points, min_digits, dim=2, min_length=1, precision=0):
    def extend_existing_boundaries(existing_solution, new_blue_points, red_points, unsafe_points, min_digits):
        unsafe_points = [Point(points=[unsafe_pt[i] for i in range(dim)]) for unsafe_pt in unsafe_points]
        for blue_pt in new_blue_points:
            for tree in existing_solution:
                box = tree.box
                if not box.contains(blue_pt):
                    tmp_box = Box2D([Interval(min(box.bs[i].lb, blue_pt.points[i]),
                                              max(box.bs[i].ub, blue_pt.points[i] + min_digits)) for i in
                                     range(len(box.bs))])
                    fprs = [b.box.fpr for b in MDP.set_cover_result if b.box != tree.box]
                    fpr = false_positive_rate(tmp_box, red_points, unsafe_points)
                    fprs.append(fpr)
                    avg_fpr = sum(fprs) / len(fprs)
                    if avg_fpr <= (1 - precision):
                        box.bs = tmp_box.bs
                        break
        return existing_solution

    uncovered_unsafe_points = [Point(points=[pt[i] for i in range(dim)]) for pt in uncovered_unsafe_points]
    safe_points = [Point(points=[pt[i] for i in range(dim)]) for pt in safe_points]

    if MDP.set_cover_result:
        existing_solution = extend_existing_boundaries(MDP.set_cover_result, uncovered_unsafe_points, safe_points,
                                                       unsafe_points, min_digits)
        uncovered_unsafe_points = [pt for pt in uncovered_unsafe_points if
                                   not any(t.box.contains(pt) for t in existing_solution)]

        MDP.abstract_state = {}
        MDP.labelling_function = {label: [] for label in MDP.ap}
        MDP.state_num = 0
        if not uncovered_unsafe_points:
            bounds = [[r.box.bs[i].lb for r in existing_solution] + [r.box.bs[i].ub for r in existing_solution] for i in
                      range(dim)]
            return existing_solution, bounds
        else:
            uncovered_unsafe_points = [Point(points=[unsafe_pt[i] for i in range(dim)]) for unsafe_pt in unsafe_points]

    domain = Box2D([Interval(0, MDP.nrow), Interval(0, MDP.ncol)])

    bounded_domain = bound(domain, uncovered_unsafe_points, min_digits=min_digits)
    root = BranchAndBoundTree(bounded_domain, children=[])

    result = expand_tree_bfs(root, uncovered_unsafe_points, safe_points, min_digits, precision=precision, min_length=min_length)

    confidence_interval = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}

    bounds = [[r.box.bs[i].lb for r in result] + [r.box.bs[i].ub for r in result] for i in range(dim)]

    return result, bounds

def set_cover(MDP, history, min_digits, dim=2, precision=0):
    def box(results, label, dim):
        for r in results:
            state = []
            for i in range(dim):
                state.extend([r.box.bs[i].lb, r.box.bs[i].ub])
            MDP.labelling_function[label].append(MDP.state_num)
            MDP.abstract_state[MDP.state_num] = [state]
            MDP.state_num += 1

    def generate_states(box_bounds):
        states = [[box_bounds[0][i], box_bounds[0][i + 1]] for i in range(len(box_bounds[0]) - 1)]
        for d in range(1, len(box_bounds)):
            states = [s + [box_bounds[d][i], box_bounds[d][i + 1]] for s in states for i in
                      range(len(box_bounds[d]) - 1)]
        return states

    def is_overlapping(state1, states):
        for state2 in states:
            if state1[0] >= state2[1] or state1[1] <= state2[0] or state1[3] <= state2[2] or state1[2] >= state2[3]:
                return False
        return True

    df_safe = history[history['label'] != "unsafe"]
    df_safe = df_safe[~df_safe['sprime'].astype(str).duplicated()]
    df_unsafe = history[history['label'] == "unsafe"]
    df_unsafe = df_unsafe[~df_unsafe['sprime'].astype(str).duplicated()]

    safe_points = np.array(df_safe['sprime'])
    unsafe_points = np.array(df_unsafe['sprime'])

    # target_points = numpy.array(df_goal['s'])
    if MDP.set_cover_result:
        uncovered_unsafe_points = []
        # uncovered_target_points = []
        for p in unsafe_points:
            if MDP.get_label(MDP.get_state(p)) != 'unsafe':
                uncovered_unsafe_points.append(p)

        uncovered_unsafe_points = np.array(uncovered_unsafe_points)

    else:
        uncovered_unsafe_points = unsafe_points

    if uncovered_unsafe_points.size == 0:
        return None

    results, bounds = branchandbound(MDP=MDP, uncovered_unsafe_points=uncovered_unsafe_points, safe_points=safe_points,
                                          unsafe_points=unsafe_points, min_digits=min_digits, dim=dim,
                                          precision=precision)

    MDP.set_cover_result = results

    box_bounds = [[0, MDP.nrow], [0, MDP.ncol]]
    box_bounds = [box_bounds[i] + bounds[i] for i in range(dim)]

    box(results, "unsafe", dim=dim)
    box_bounds = [sorted(list(set(box_bounds[i]))) for i in range(dim)]

    states = generate_states(box_bounds)

    for state in states:
        if any(is_overlapping(state, v) for v in MDP.abstract_state.values()):
            continue
        MDP.labelling_function['safe'].append(MDP.state_num)
        MDP.abstract_state[MDP.state_num] = [state]
        MDP.state_num += 1

    MDP.box_bounds = [sorted(list(set(box_bounds[i] + MDP.box_bounds[i]))) for i in range(dim)]
    return results

def build_abstraction(MDP, cfg):
    def sample_size(box_num, epsilon, delta):
        # compute the sample size for bounding prediction error at most epsilon with a high probability in a learning function with s unions of boxes
        d = 4  # VC Dimension of axis-aligned rectangles is 4
        VC = 2 * d * box_num * math.log(3 * box_num)
        magnitude1 = 4 / epsilon * math.log(2 / delta)
        magnitude2 = (8 * VC) / epsilon * math.log(13 / epsilon)
        sample_min = math.ceil(max(magnitude1, magnitude2))
        return sample_min

    def merge(MDP, epsilon=0.01):
        merge_dict = {}
        for label, states in MDP.labelling_function.items():
            for state in states:
                if state not in MDP.labelling_matrix:
                    continue
                possible_states = MDP.possible_matrix[state].intersection(set(states)).difference({state})

                merged_states = [s for s in possible_states
                                 if s in states and all(
                        abs(MDP.labelling_matrix[state][action][l] - MDP.labelling_matrix[s][action][l]) <= epsilon for
                        action in range(MDP.MDP.action_space.n) for l in MDP.labelling_function.keys())]

                if merged_states:
                    merge_dict[state] = merged_states
                    for s in merge_dict[state]:
                        if state not in MDP.abstract_state:
                            MDP.abstract_state[state] = []
                        for bound in MDP.abstract_state[s]:
                            MDP.abstract_state[state].append(bound)
                        MDP.abstract_state.pop(s)
                        MDP.labelling_function[label].remove(s)

        return merge_dict

    if not MDP.set_cover_result:
        history = pandas.DataFrame(MDP.history)
    else:
        box_num = len(MDP.set_cover_result)
        estimated_sample_num = sample_size(box_num=box_num, epsilon=1 - cfg.hyper.precision,
                                           delta=1 - cfg.hyper.confidence)
        if len(MDP.history["s"]) > estimated_sample_num:
            history = pandas.DataFrame(MDP.history).sample(n=estimated_sample_num, random_state=1)
        else:
            history = pandas.DataFrame(MDP.history)

    min_digits = 1 if cfg.env.state_type == "discrete" else 10 ** (-cfg.hyper.rounding_digits)

    precision = cfg.hyper.precision

    # merge epsilon-similar abstract states
    epsilon = cfg.hyper.u

    MDP.set_cover_result = None
    MDP.state_num = 0
    MDP.labelling_function = {label: [] for label in MDP.ap}
    MDP.abstract_state = {}
    set_cover(MDP, history, dim=2, min_digits=min_digits, precision=precision)

    result = True

    MDP.prob_matrix(history=history, rounding_digits=cfg.hyper.rounding_digits)

    while result:
        result = merge(MDP, epsilon=epsilon)
        MDP.prob_matrix(history=history, rounding_digits=cfg.hyper.rounding_digits)

    print("Constructed an abstract MDP with {} abstract state space.".format(MDP.state_num))

    if hasattr(MDP, "flag"):
        unsafe_states = [str([s, flag]) for s in MDP.labelling_function["unsafe"] for flag in
                         [0, 1]]
    else:
        unsafe_states = [str(s) for s in MDP.labelling_function["unsafe"]]

    MDP.relevant_states = unsafe_states
    MDP.deadlock_states = set()
    print("Safety relevant abstract states: ", MDP.relevant_states)

def bayes_hypo_test(violates, prob_lambda, alpha, beta, rounding_digits=3):
    prior_p_h0 = round(1 - stats.beta.cdf(prob_lambda, alpha, beta), rounding_digits)
    prior_p_h1 = round(1 - prior_p_h0, rounding_digits)
    prior = round(prior_p_h1 / prior_p_h0, rounding_digits) if prior_p_h0 > 0 else 1

    alpha_prime = alpha + violates.count(0)
    beta_prime = beta + len(violates) - violates.count(0)

    mu = alpha_prime / (alpha_prime + beta_prime)
    alpha_prime = round(mu, rounding_digits)
    beta_prime = round(1 - mu, rounding_digits)

    cdf_post = round(stats.beta.cdf(prob_lambda, alpha_prime, beta_prime), rounding_digits)
    if cdf_post != 0 and not np.isnan(cdf_post):
        estimation = round(prior * ((1 / cdf_post) - 1), rounding_digits)
    else:
        estimation = 1e6
    return estimation, alpha_prime, beta_prime

def simulation_learning(MDP, cfg, counterexample, Q):
    while MDP.sim_episode_num < cfg.hyper.sim_episode_num:
        done = False
        state = MDP.get_position(MDP.reset())
        while not done:
            abstract_state = MDP.get_state(state)
            action = list(counterexample[abstract_state].keys())[0]
            obs, reward, done, info = MDP.step(action)
            next_state = MDP.get_position(obs)
            Q[str(state)][action] = (1 - cfg.hyper.alpha) * Q[str(state)][action] + cfg.hyper.alpha * (
                    reward + cfg.hyper.gamma * max(Q[str(next_state)].values()))
            # Q[str(state)][action] = (1 - cfg.hyper.alpha) * Q[str(state)][action] + cfg.hyper.alpha * (
            #         reward + cfg.hyper.gamma * max(Q[str(next_state)].values()))
            state = next_state

    return Q

def cex_guided_learning(MDP, cfg, target_state, blocking_sets, previous_cex={}, Q=None, callback=None):
    prob_lambda = cfg.env.failureprob
    initial_concrete_state = MDP.reset()
    initial_state = MDP.get_state(MDP.get_position(initial_concrete_state))
    print("Safety requirement: P [F unsafe] <= {} ".format(prob_lambda))

    cex, variable_results = minsetCounterexampleCVX(mdp=MDP.transition_matrix, init=initial_state,
                                                        target={target_state},
                                                        lambda_threshold=prob_lambda, blocking_sets=blocking_sets)

    if cex:
        MDP.trace = []
        MDP.policy = cex
        for state, transitions in cex.items():
            for action, next_state in transitions.items():
                for next_s in next_state:
                    previous_cex[f'{state}_{action}_{next_s}'] += 1
                    if next_s in target_state or next_s in MDP.deadlock_states:
                        blocking_sets.add(f'{state}_{action}_{next_s}')

        MDP.sim_episode_num = 0
        if cfg.alg.algorithm == 'QLearning':
            Q = simulation_learning(MDP, cfg, cex, Q)
        else:
            while MDP.sim_episode_num < cfg.hyper.sim_episode_num:
                collect_rollouts(MDP, callback, guidance=True, sim_episode_num=cfg.hyper.sim_episode_num)
                MDP.model.train(batch_size=MDP.model.batch_size, gradient_steps=MDP.model.gradient_steps)

    return cex, Q, blocking_sets, previous_cex

def collect_rollouts(MDP, callback, sim_episode_num=0, log_interval=4, guidance=False):
    MDP.model.policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0

    callback.on_rollout_start()
    continue_training = True

    while num_collected_steps < MDP.model.train_freq.frequency:
        if guidance:
            s = MDP.get_state(MDP.s)
            actions = list(MDP.policy[s].keys())
            buffer_actions = np.array(actions)
        else:
            # Select action randomly or according to policy
            actions, buffer_actions = MDP.model._sample_action(MDP.model.learning_starts)

        # Rescale and perform action
        new_obs, rewards, dones, infos = MDP.model.env.step(actions)

        MDP.model.num_timesteps += MDP.model.env.num_envs
        num_collected_steps += 1

        # Give access to local variables
        callback.update_locals(locals())
        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(num_collected_steps * MDP.model.env.num_envs, num_collected_episodes, continue_training=False)

        # Retrieve reward and episode length if using Monitor wrapper
        MDP.model._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        MDP.model._store_transition(MDP.model.replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

        MDP.model._update_current_progress_remaining(MDP.model.num_timesteps, MDP.model._total_timesteps)

        MDP.model._on_step()

        for idx, done in enumerate(dones):
            if done:
                if not guidance:
                    MDP.model._episode_num += 1

                num_collected_episodes += 1

                # Log training infos
                if not guidance and log_interval is not None and MDP.model._episode_num % log_interval == 0:
                    MDP.model._dump_logs()
    callback.on_rollout_end()

    return RolloutReturn(num_collected_steps * MDP.model.env.num_envs, num_collected_episodes, continue_training)

def train(cfg):
    #Enviorment Setup
    if "FrozenLake" in cfg.env.MDP:
        env = gym.make(cfg.env.MDP, is_slippery=cfg.env.is_slippery)
    elif "grid" in cfg.env.MDP:
        env = gym.make(cfg.env.MDP, mode=cfg.env.mode)
    else:
        env = gym.make(cfg.env.MDP)

    MDP = Abstraction(env=env, cfg=cfg)

    action_space = range(MDP.action_space.n)
    prob_lambda = cfg.env.failureprob
    alpha_bayes = cfg.hyper.alpha_bayes
    beta_bayes = cfg.hyper.beta_bayes
    checked_episode = 0
    decayed_random_prob = cfg.hyper.epsilon

    history = pandas.DataFrame({"s": [], "action": [], "sprime": [], "label": []})

    Q = {}

    MDP.reset()

    print("Created an MDP with gym environment: {}".format(cfg.env.MDP))

    if cfg.alg.algorithm == 'QLearning':
        for episode in tqdm(range(cfg.hyper.episode_num)):
            if (cfg.alg.guidance and (MDP.episode - checked_episode) > cfg.hyper.safety_check_num):
                hypo_estimation, alpha_bayes, beta_bayes = bayes_hypo_test(
                    MDP.violates[MDP.offline_episode:MDP.episode],
                    prob_lambda=prob_lambda,
                    alpha=alpha_bayes,
                    beta=beta_bayes)
                checked_episode = int(MDP.episode)

                if hypo_estimation > cfg.hyper.bayes_factor:
                    print(
                        "==================================================================================================")
                    print("Offline learning phase has been triggered with the Bayes Hypothesis B = {}".format(
                        hypo_estimation))
                    MDP.simulation = True
                    alpha_bayes = cfg.hyper.alpha_bayes
                    beta_bayes = cfg.hyper.beta_bayes
                    MDP.offline_episode = MDP.episode
                    build_abstraction(MDP, cfg)
                    for target_state in MDP.relevant_states:
                        cex = cfg.alg.guidance
                        cex_num = 0
                        blocking_sets = set()
                        previous_cex = defaultdict(int)
                        while cex:
                            cex, Q, blocking_sets, previous_cex = cex_guided_learning(MDP, cfg,
                                                                                      target_state=target_state,
                                                                                      blocking_sets=blocking_sets,
                                                                                      previous_cex=previous_cex, Q=Q)
                            cex_num += 1 if cex else 0
                    MDP.simulation = False
                    print(
                        "=================================================================================================")

            decayed_random_prob *= cfg.hyper.epsilon_decay

            current_state = MDP.get_position(MDP.reset())

            if str(current_state) not in Q:
                Q[str(current_state)] = {action: 0 for action in action_space}

            iteration = 0
            done = False

            while iteration < cfg.hyper.iteration_num and not done:
                iteration += 1

                # epsilon-greedy exploration
                if np.random.rand() < decayed_random_prob:
                    policy = {action: 1 for action in action_space}
                else:
                    policy = {action: (value - min(Q[str(current_state)].values())) / (
                                max(Q[str(current_state)].values()) - min(Q[str(current_state)].values())) if max(
                        Q[str(current_state)].values()) - min(Q[str(current_state)].values()) != 0 else 1 / len(
                        action_space) for action, value in Q[str(current_state)].items() if
                              abs(value - max(Q[str(current_state)].values())) < 1e-10}

                current_action = np.random.choice(list(policy.keys()),
                                                  p=[v / sum(policy.values()) for v in policy.values()])

                obs, reward, done, info = MDP.step(current_action)

                next_state = MDP.get_position(obs)

                if str(next_state) not in Q:
                    Q[str(next_state)] = {action: 0 for action in action_space}

                Q[str(current_state)][current_action] = (1 - cfg.hyper.alpha) * Q[str(current_state)][
                    current_action] + cfg.hyper.alpha * (reward + cfg.hyper.gamma * max(
                    Q[str(next_state)].values()))

                current_state = next_state

            if MDP.episode % 50 == 0 and len(MDP.violates) > 0:
                safety_rate = round(MDP.violates.count(1) / len(MDP.violates), 5)
                print("Safety rate at episode {}/{}: {}".format(episode, cfg.hyper.episode_num, safety_rate))
    elif cfg.alg.algorithm == 'DQN':
        from stable_baselines3 import DQN

        MDP.model = DQN("MlpPolicy", MDP, verbose=1, learning_rate=cfg.hyper.learning_rate,
                        learning_starts=cfg.hyper.learning_starts, train_freq=(cfg.hyper.train_freq, "step"),
                        max_grad_norm=cfg.hyper.max_grad_norm, target_update_interval=cfg.hyper.n_steps,
                        exploration_fraction=cfg.hyper.exploration_fraction,
                        exploration_initial_eps=cfg.hyper.exploration_initial_eps,
                        exploration_final_eps=cfg.hyper.exploration_final_eps, gamma=cfg.hyper.gamma,
                        batch_size=cfg.hyper.batch_size, gradient_steps=cfg.hyper.gradient_steps)

        total_timesteps, callback = MDP.model._setup_learn(total_timesteps=cfg.hyper.total_timesteps)
        callback.on_training_start(locals(), globals())
        log_interval = 4

        while MDP.episode < cfg.hyper.episode_num:
            if cfg.alg.guidance and (MDP.episode - checked_episode) >= cfg.hyper.safety_check_num:
                hypo_estimation, alpha_bayes, beta_bayes = bayes_hypo_test(
                    MDP.violates[MDP.offline_episode:MDP.episode],
                    prob_lambda=prob_lambda,
                    alpha=alpha_bayes,
                    beta=beta_bayes)
                checked_episode = MDP.episode
                if hypo_estimation > cfg.hyper.bayes_factor:
                    print(
                        "==================================================================================================")
                    print("Offline learning phase has been triggered with the Bayes Hypothesis B = {}".format(
                        hypo_estimation))
                    MDP.simulation = True
                    alpha_bayes = cfg.hyper.alpha_bayes
                    beta_bayes = cfg.hyper.beta_bayes
                    MDP.offline_episode = MDP.episode
                    build_abstraction(MDP, cfg)
                    for state, action_state in MDP.concrete_transition_history.items():
                        for action, next_state_dist in action_state.items():
                            if len(next_state_dist) > cfg.hyper.offline_rolling_window:
                                possible_next_states = list(next_state_dist.keys())
                                subsample = set(random.sample(possible_next_states, cfg.hyper.offline_rolling_window))
                                MDP.concrete_transition_history[state][action] = {k: v for k, v in
                                                                                  next_state_dist.items() if
                                                                                  k in subsample}
                    for target_state in MDP.relevant_states:
                        cex = cfg.alg.guidance
                        cex_num = 0
                        blocking_sets = set()
                        previous_cex = defaultdict(int)
                        while cex:
                            cex, Q, blocking_sets, previous_cex = cex_guided_learning(MDP, cfg,
                                                                                      target_state=target_state,
                                                                                      blocking_sets=blocking_sets,
                                                                                      previous_cex=previous_cex, callback=callback)
                            cex_num += 1 if cex else 0
                    MDP.simulation = False
                    gradient_steps = MDP.model.gradient_steps
                    MDP.model.train(batch_size=MDP.model.batch_size, gradient_steps=gradient_steps)
                    print(
                        "==================================================================================================")

            rollout = collect_rollouts(MDP, callback=callback)

            if rollout.continue_training is False:
                break

            if MDP.model.num_timesteps > 0 and MDP.model.num_timesteps > MDP.model.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = MDP.model.gradient_steps if MDP.model.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    MDP.model.train(batch_size=MDP.model.batch_size, gradient_steps=gradient_steps)

            if MDP.episode % 100 == 0 and len(MDP.violates) > 0:
                safety_rate = round(MDP.violates.count(1) / len(MDP.violates), 5)
                print("Safety rate at episode {}/{}: {}".format(MDP.episode, cfg.hyper.episode_num, safety_rate))

        callback.on_training_end()
    return MDP