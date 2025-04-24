import numpy as np
from cluster_base import ClusterMatchingBase

class ClusterMatching(ClusterMatchingBase):
    def match_all(self):
        self.transitions = []
        for i in range(len(self.formations) - 1):
            fpos, tpos = self.formations[i], self.formations[i + 1]
            flab, tlab = self.labels[i], self.labels[i + 1]

            # 비용 행렬 계산
            cost_matrix = self.cost_matrix_fn(fpos, flab, tpos, tlab, self.K)

            # 클러스터 간 매칭
            row_ind, col_ind = self.cluster_matcher(cost_matrix)

            row_inds, col_inds = [], []
            for fc, tc in zip(row_ind, col_ind):
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
        print("\n[ClusterMatching] 평균 이동 거리:", np.mean(total_dists))
        print("[ClusterMatching] 최대 이동 거리:", np.max(total_dists))