import numpy as np
from scipy.optimize import linear_sum_assignment

# 클러스터 간 평균 거리 행렬 생성 (Hungarian용)
def mean_cost_matrix(fpos, flab, tpos, tlab, K):
    cost = np.zeros((K, K))
    for i in range(K):
        fi = fpos[flab == i]
        for j in range(K):
            tj = tpos[tlab == j]
            cost[i, j] = np.mean(np.linalg.norm(fi[:, None] - tj[None, :], axis=2))
    return cost

# 클러스터 간 최대 거리 행렬 생성 (LBAP용)
def bottleneck_cost_matrix(fpos, flab, tpos, tlab, K):
    cost = np.zeros((K, K))
    for i in range(K):
        fi = fpos[flab == i]
        for j in range(K):
            tj = tpos[tlab == j]
            cost[i, j] = np.max(np.linalg.norm(fi[:, None] - tj[None, :], axis=2))
    return cost

# 에이전트 간 Hungarian 매칭 (sum of distances 최소화)
def hungarian_match(A, B):
    cost_matrix = np.linalg.norm(A[:, None] - B[None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind

# 에이전트 간 LBAP 매칭 (max of distances 최소화)
def bottleneck_match(A, B):
    cost_matrix = np.linalg.norm(A[:, None] - B[None, :], axis=2)
    low, high = 0, np.max(cost_matrix)

    def can_assign(thresh):
        mask = cost_matrix <= thresh
        try:
            row_ind, col_ind = linear_sum_assignment(~mask)
            feasible = np.all(mask[row_ind, col_ind])
            return feasible, (row_ind, col_ind)
        except:
            return False, None

    best = None
    for _ in range(50):
        mid = (low + high) / 2
        feasible, matching = can_assign(mid)
        if feasible:
            high = mid
            best = matching
        else:
            low = mid

    return best

# 클러스터 간 LBAP 매칭 (최대 거리 최소화)
def bottleneck_cluster_assignment(cost_matrix):
    low, high = 0, np.max(cost_matrix)

    def is_feasible(thresh):
        mask = cost_matrix <= thresh
        try:
            row_ind, col_ind = linear_sum_assignment(~mask)
            feasible = np.all(mask[row_ind, col_ind])
            return feasible, (row_ind, col_ind)
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
