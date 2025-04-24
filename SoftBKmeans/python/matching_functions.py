import numpy as np
from scipy.optimize import linear_sum_assignment

# --- Cost Functions ---
def euclidean_distance_cost(A, B):
    return np.linalg.norm(A[:, None] - B[None, :], axis=2)

def cluster_mean_cost(cost_fn, fpos, flab, tpos, tlab, K):
    cost = np.zeros((K, K))
    for i in range(K):
        fi = fpos[flab == i]
        for j in range(K):
            tj = tpos[tlab == j]
            cost[i, j] = np.mean(cost_fn(fi, tj))
    return cost

def cluster_max_cost(cost_fn, fpos, flab, tpos, tlab, K):
    cost = np.zeros((K, K))
    for i in range(K):
        fi = fpos[flab == i]
        for j in range(K):
            tj = tpos[tlab == j]
            cost[i, j] = np.max(cost_fn(fi, tj))
    return cost

# --- Assignment Functions ---
def hungarian_assign(cost_matrix):
    return linear_sum_assignment(cost_matrix)

def bottleneck_assign(cost_matrix):
    low, high = 0, np.max(cost_matrix)

    def is_feasible(thresh):
        mask = cost_matrix <= thresh
        try:
            r, c = linear_sum_assignment(~mask)
            return np.all(mask[r, c]), (r, c)
        except:
            return False, None

    best = None
    for _ in range(50):
        mid = (low + high) / 2
        feasible, match = is_feasible(mid)
        if feasible:
            high = mid
            best = match
        else:
            low = mid

    return best

# --- MILP with PuLP ---
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

def solve_cluster_milp(cost_matrices):
    T = len(cost_matrices)
    K = cost_matrices[0].shape[0]

    prob = LpProblem("MultiStepClusterMatching", LpMinimize)

    x = [[[LpVariable(f"x_{t}_{i}_{j}", cat=LpBinary)
           for j in range(K)] for i in range(K)] for t in range(T)]

    prob += lpSum(cost_matrices[t][i][j] * x[t][i][j]
                  for t in range(T) for i in range(K) for j in range(K))

    for t in range(T):
        for i in range(K):
            prob += lpSum(x[t][i][j] for j in range(K)) == 1
        for j in range(K):
            prob += lpSum(x[t][i][j] for i in range(K)) == 1

    prob.solve()

    matchings = []
    for t in range(T):
        match = []
        for i in range(K):
            for j in range(K):
                if x[t][i][j].varValue > 0.5:
                    match.append((i, j))
        matchings.append(match)

    return matchings

def multi_step_cluster_matcher_pulp(formations, labels, cost_fn):
    T = len(formations) - 1
    K = len(np.unique(labels[0]))
    cost_matrices = []

    for t in range(T):
        fpos, flab = formations[t], labels[t]
        tpos, tlab = formations[t + 1], labels[t + 1]
        cost_matrix = cost_fn(fpos, flab, tpos, tlab, K)
        cost_matrices.append(cost_matrix)

    return solve_cluster_milp(cost_matrices)
