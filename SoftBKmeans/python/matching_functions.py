
import numpy as np
from scipy.optimize import linear_sum_assignment

# Cost functions
def mean_cost(A, B):
    return np.linalg.norm(A[:, None] - B[None, :], axis=2)

def max_cost(A, B):
    return np.linalg.norm(A[:, None] - B[None, :], axis=2)

def cluster_mean_cost(fpos, flab, tpos, tlab, K):
    cost = np.zeros((K, K))
    for i in range(K):
        fi = fpos[flab == i]
        for j in range(K):
            tj = tpos[tlab == j]
            cost[i, j] = np.mean(mean_cost(fi, tj))
    return cost

def cluster_max_cost(fpos, flab, tpos, tlab, K):
    cost = np.zeros((K, K))
    for i in range(K):
        fi = fpos[flab == i]
        for j in range(K):
            tj = tpos[tlab == j]
            cost[i, j] = np.max(max_cost(fi, tj))
    return cost

# Assign functions
def hungarian_assign(cost_matrix=None, A=None, B=None):
    if cost_matrix is None and A is not None and B is not None:
        cost_matrix = mean_cost(A, B)
    return linear_sum_assignment(cost_matrix)

def bottleneck_assign(cost_matrix=None, A=None, B=None):
    if cost_matrix is None and A is not None and B is not None:
        cost_matrix = max_cost(A, B)
    low, high = 0, np.max(cost_matrix)

    def is_feasible(thresh):
        mask = cost_matrix <= thresh
        try:
            row_ind, col_ind = linear_sum_assignment(~mask)
            return np.all(mask[row_ind, col_ind]), (row_ind, col_ind)
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