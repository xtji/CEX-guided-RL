import cvxpy
from typing import Dict, List, Set, Union
from sympy import Rational


def minProb0(mdp: Dict[int, Dict[str, Dict[int, Rational]]], target: Set[int]):
    S = set(mdp.keys())
    R = target.copy()
    while True:
        R_prime = R.copy()
        for s in mdp:
            reaches_R_Prime_via_every_actions = True

            for action in mdp[s]:
                reachable_via_action = set(mdp[s][action].keys())
                if len(R_prime.intersection(reachable_via_action)) == 0:
                    reaches_R_Prime_via_every_actions = False
                    break
            if reaches_R_Prime_via_every_actions:
                R.add(s)
        if R_prime == R:
            break
    return S.difference(R)

def minProb1(mdp: Dict[int, Dict[str, Dict[int, Rational]]], target: Set[int], minProb0set: Set[int] = None):
    if minProb0set is None:
        minProb0set = minProb0(mdp, target)

    S = set(mdp.keys())
    R = S.difference(minProb0set)

    # TO be continued
    while True:
        R_prime = R.copy()
        for s in R_prime:
            reachable_R_Prime_via_any_action = False

            for action in mdp[s]:
                reachable_via_action = set(mdp[s][action].keys())
                reachable_via_action_not_in_R_prime = reachable_via_action.difference(
                    R_prime)
                if len(R_prime.intersection(reachable_via_action_not_in_R_prime)) > 0:
                    reachable_R_Prime_via_any_action = True
                    break
            if reachable_R_Prime_via_any_action:
                R.add(s)
        if R_prime == R:
            break
    return R

def maxProb0(mdp: Dict[int, Dict[str, Dict[int, Rational]]], target: Set[int]):
    S = set(mdp.keys())
    R = target.copy()
    while True:
        R_prime = R.copy()

        for s in mdp:
            reaches_R_Prime_via_any_action = False

            for action in mdp[s]:
                reachable_via_action = set(mdp[s][action].keys())
                if len(R_prime.intersection(reachable_via_action)) > 0:
                    reaches_R_Prime_via_any_action = True
                    break
            if reaches_R_Prime_via_any_action:
                R.add(s)
        if R_prime == R:
            break
    return S.difference(R)

def maxProb1(mdp: Dict[int, Dict[str, Dict[int, Rational]]], target: Set[int]):
    S = set(mdp.keys())
    R = S.copy()

    while True:
        R_prime = R.copy()
        R = target.copy()

        while True:
            R_prime_prime = R.copy()

            for s in mdp:
                for action in mdp[s]:
                    reachable_via_action = set(mdp[s][action].keys())

                    all_reach_R_prime = len(reachable_via_action.difference(R_prime)) == 0
                    any_reaches_R_prime_prime = len(R_prime_prime.intersection(reachable_via_action)) > 0

                    if all_reach_R_prime and any_reaches_R_prime_prime:
                        R.add(s)

            if R_prime_prime == R:
                break

        if R_prime == R:
            break
    return R

def minsetCounterexampleCVX(mdp: Dict[int, Dict[str, Dict[int, Rational]]], init: str, target: Set[str],
                       lambda_threshold: Union[float, Rational], blocking_sets: Set[str] = set(), solver='GUROBI', previous_results={}, weights_function={}):
    constraints = []
    domain_constraints = []

    state_to_cvxpy_var = {}
    labels = {}

    def to_cvxpy_var(name: str, is_int: bool = False):
        if name not in state_to_cvxpy_var:
            state_to_cvxpy_var[name] = cvxpy.Variable(name=name, integer=True) if is_int else cvxpy.Variable(name=name)
        result = state_to_cvxpy_var[name]
        return result

    # Keep track of labels to be included or not in the model and their corresponding actions
    x_init = to_cvxpy_var(f'x_{init}', is_int=False)
    constraints.append(x_init >= lambda_threshold)

    for s in mdp:
        xs = to_cvxpy_var(f'x_{s}', is_int=False)
        domain_constraints.append(xs >= 0)
        domain_constraints.append(xs <= 1)

        if s in target:
            # Eq 3.1c
            constraints.append(xs == 1)
        #elif s in transition:
        #    constraints.append(xs >= 1e-7)

        elif s in maxProb0(mdp, target):
            constraints.append(xs == 0)
        else:
            sigma = []
            for action in mdp[s]:
                # sigma is either 0 or 1, (1 if eta is chosen)
                sigma_s_eta = to_cvxpy_var(f'sigma_{s}_{action}', is_int=True)
                domain_constraints.append(sigma_s_eta >= 0)
                domain_constraints.append(sigma_s_eta <= 1)
                sigma.append(sigma_s_eta)
                p_eta = []
                xl_eta = []
                for dest_state in mdp[s][action]:
                    xd = to_cvxpy_var(f'x_{dest_state}', is_int=False)
                    #if dest_state not in the mdp, set x_dest to zero
                    if dest_state not in mdp:
                        domain_constraints.append(xs == 0)
                    p_s_eta_sprime = to_cvxpy_var(
                        f'p_{s}_{action}_{dest_state}', is_int=False)

                    domain_constraints.append(p_s_eta_sprime >= 0)
                    # Eq 3.1g
                    constraints.append(
                        p_s_eta_sprime <= mdp[s][action][dest_state] * xd)
                    p_eta.append(p_s_eta_sprime)
                    # xl in Eq 3.1a and Eq 3.1f, is a variable of label
                    xl = to_cvxpy_var(
                        f'xl_{s}_{action}_{dest_state}', is_int=True)
                    xl_eta.append(xl)
                    if f'{s}_{action}_{dest_state}' in blocking_sets:
                        domain_constraints.append(xl == 0)
                    else:
                        domain_constraints.append(xl >= 0)
                        domain_constraints.append(xl <= 1)

                    labels[xl] = (s, action)
                    # Eq 3.1f
                    constraints.append(p_s_eta_sprime <= xl)
                # Eq 3.1h
                constraints.append(xs <= (1 - sigma_s_eta) + sum(p_eta))

            # Eq 3.1d
            constraints.append(sum(sigma) <= 1)

            # Eq 3.1e
            constraints.append(xs <= sum(sigma))

    problematic_states = minProb0(mdp, target).difference(maxProb0(mdp, target))
    problematic_state_action_pairs = [(s, a) for s in problematic_states for a in mdp[s] if
                                      any(s_dest in problematic_states for s_dest in mdp[s][a].keys())]

    # problematic_state_action_pairs = [(s, a) for s in problematic_states for a in mdp[s] if set(mdp[s][a].keys()).issubset(problematic_states)]

    for s, a in problematic_state_action_pairs:
        rs = to_cvxpy_var(f'r_{s}', is_int=False)
        domain_constraints.append(rs >= 0)
        domain_constraints.append(rs <= 1)

        t_eta = []
        sigma_s_eta = to_cvxpy_var(f'sigma_{s}_{a}', is_int=True)
        for s_dest in mdp[s][a]:
            # t_s_s_prime in Eq 3.1i, Eq 3.1j (integers either 0 or 1)
            t_s_eta_sprime = to_cvxpy_var(
                f'xl_{s}_{a}_{s_dest}', is_int=True)
            domain_constraints.append(t_s_eta_sprime >= 0)
            domain_constraints.append(t_s_eta_sprime <= 1)
            t_eta.append(t_s_eta_sprime)

            r_s_prime = to_cvxpy_var(f'r_{s_dest}', is_int=False)
            domain_constraints.append(r_s_prime >= 0)
            domain_constraints.append(r_s_prime <= 1)
            # Eq 3.1j
            epsilon = 1e-3
            constraints.append(
                rs + epsilon <= r_s_prime + (1 - t_s_eta_sprime)
            )
        # Eq 3.1i
        constraints.append(sigma_s_eta <= sum(t_eta))

    constraints = constraints + domain_constraints

    if not weights_function:
        problem = cvxpy.Problem(cvxpy.Minimize(-1 * x_init + sum(labels.keys())), constraints)
    else:
        weights = {}
        min_weight = 1
        for variable in labels.keys():
            _, state, action, next_state = variable.name().split("_")
            weights[variable] = weights_function[f'{state}_{action}_{next_state}'] if f'{state}_{action}_{next_state}' in weights_function else 0.5
        objective = - 0.5 * min_weight * x_init + sum(weights[i]*i for i in labels.keys())
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    if previous_results:
        for var in problem.variables():
            var.value = previous_results[var.name()]
        warm_start = True
    else:
        warm_start = False

    result = {}
    solvers = cvxpy.installed_solvers()
    if solver not in solvers:
        solver = 'SCIP' if 'SCIP' in solvers else 'GLPK_MI'

    try:
        if solver == 'GUROBI':
            problem.solve(solver=cvxpy.GUROBI, verbose=False, TimeLimit= 30, Threads=4)
        elif solver == 'SCIP':
            problem.solve(solver=cvxpy.SCIP, verbose=False, scip_params={"limits/time": 60})
        else:
            problem.solve(solver=solver)
    except cvxpy.error.SolverError:
        problem.solve(solver='GLPK_MI')
    except AttributeError:
        return result, previous_results
    #print(problem.status)
    if problem.status == 'optimal':
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Solve time:", problem.solver_stats.solve_time)
        variable_results = {variable.name(): variable.value for variable in problem.variables() if variable.value > 0}
        previous = []
        sprimes = []

        for variable in reversed(problem.variables()):
            # rounding to 7 decimal digits to account for numerical error
            if (variable in labels.keys() and variable.value == 1):
                _, state, action, next_state = variable.name().split("_")
                if action.isdigit():
                    action = int(action)
                if state not in result:
                    result[state] = {}
                if action not in result[state]:
                    result[state][action] = []
                result[state][action].append(next_state)
                if next_state in target:
                    previous.append([state, action, next_state])
                sprimes.append(next_state)
                print(f'In state {state} take action {action} because of transition {variable}')

        if all(t not in sprimes for t in target):
            result = {}
        return result, variable_results
    else:
        print('Cannot find a solution for current problem for target state', target)
        return result, previous_results