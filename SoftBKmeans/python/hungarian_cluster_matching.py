
import numpy as np
from scipy.optimize import linear_sum_assignment
from cluster_base import ClusterMatchingBase

class HungarianClusterMatching(ClusterMatchingBase):
    def match_all(self):
        self.transitions = []
        for i in range(len(self.formations) - 1):
            fpos, tpos = self.formations[i], self.formations[i + 1]
            flab, tlab = self.labels[i], self.labels[i + 1]

            cost_matrix = self.cluster_matcher(fpos, flab, tpos, tlab, self.K)
            cmatch = list(zip(*linear_sum_assignment(cost_matrix)))

            row_inds, col_inds = [], []
            for fc, tc in cmatch:
                fidx = np.where(flab == fc)[0]
                tidx = np.where(tlab == tc)[0]
                A, B = fpos[fidx], tpos[tidx]
                r, c = self.agent_matcher(A, B)
                row_inds.extend(fidx[r])
                col_inds.extend(tidx[c])

            self.transitions.append((np.array(row_inds), np.array(col_inds)))

    def summarize(self):
        total_dists = np.zeros(self.formations[0].shape[0])
        for i, (rind, cind) in enumerate(self.transitions):
            moved = np.linalg.norm(self.formations[i][rind] - self.formations[i + 1][cind], axis=1)
            total_dists += moved
        print("\n[HungarianClusterMatching] 평균 이동 거리:", np.mean(total_dists))
        print("[HungarianClusterMatching] 최대 이동 거리:", np.max(total_dists))