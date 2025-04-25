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