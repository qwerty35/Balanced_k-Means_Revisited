import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary

class MultiStepClusterMatching:
    def __init__(self, formation_files, num_clusters, clustering_params,
                 cluster_cost_fn, cluster_assign_fn,
                 agent_cost_fn, agent_assign_fn):
        self.files = formation_files
        self.K = num_clusters
        self.params = clustering_params
        self.cluster_cost_fn = cluster_cost_fn
        self.cluster_assign_fn = cluster_assign_fn
        self.agent_cost_fn = agent_cost_fn
        self.agent_assign_fn = agent_assign_fn

        self.formations = []
        self.labels = []
        self.transitions = []

    def run(self):
        self.cluster_formations()
        self.match_all()
        self.visualize_transitions()
        self.summarize()

    def cluster_formations(self):
        from balkmeans import balkmeans
        for path in self.files:
            data = np.loadtxt(path)
            labels = balkmeans(data.tolist(), **self.params)
            self.formations.append(data)
            self.labels.append(np.array(labels))

    def match_all(self):
        cost_fn = lambda f, fl, t, tl, K: self.cluster_cost_fn(self.agent_cost_fn, f, fl, t, tl, K)
        cluster_matches = self.multi_step_cluster_matcher_pulp(self.formations, self.labels, cost_fn)
        self.transitions = []

        for t, step in enumerate(cluster_matches):
            row_inds, col_inds = [], []
            for fc, tc in step:
                fidx = np.where(self.labels[t] == fc)[0]
                tidx = np.where(self.labels[t + 1] == tc)[0]
                A, B = self.formations[t][fidx], self.formations[t + 1][tidx]
                cost_matrix = self.agent_cost_fn(A, B)
                r, c = self.agent_assign_fn(cost_matrix)
                row_inds.extend(fidx[r])
                col_inds.extend(tidx[c])
            self.transitions.append((np.array(row_inds), np.array(col_inds)))

    def solve_cluster_milp(self, cost_matrices):
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

    def multi_step_cluster_matcher_pulp(self, formations, labels, cost_fn):
        T = len(formations) - 1
        K = len(np.unique(labels[0]))
        cost_matrices = []

        for t in range(T):
            fpos, flab = formations[t], labels[t]
            tpos, tlab = formations[t + 1], labels[t + 1]
            cost_matrix = cost_fn(fpos, flab, tpos, tlab, K)
            cost_matrices.append(cost_matrix)

        return self.solve_cluster_milp(cost_matrices)

    def summarize(self):
        total_dists = np.zeros(self.formations[0].shape[0])
        for i, (rind, cind) in enumerate(self.transitions):
            moved = np.linalg.norm(self.formations[i][rind] - self.formations[i + 1][cind], axis=1)
            print(f"[Transition {i} → {i + 1}] 평균 이동 거리: {np.mean(moved):.2f}, 최대 이동 거리: {np.max(moved):.2f}")
            total_dists += moved
        print(f"\n[Total] 평균 이동 거리: {np.mean(total_dists):.2f}")
        print(f"[Total] 최대 이동 거리: {np.max(total_dists):.2f}")

    def visualize_transitions(self):
        import matplotlib.pyplot as plt
        for i, (rind, cind) in enumerate(self.transitions):
            plt.figure(figsize=(10, 10))
            for a, b in zip(rind, cind):
                plt.plot(
                    [self.formations[i][a, 0], self.formations[i + 1][b, 0]],
                    [self.formations[i][a, 1], self.formations[i + 1][b, 1]],
                    color=plt.cm.tab20(self.labels[i][a] % 20), alpha=0.2, linewidth=0.5
                )
            plt.scatter(self.formations[i][:, 0], self.formations[i][:, 1], c=self.labels[i], cmap='tab20', s=5, label='From')
            plt.scatter(self.formations[i + 1][cind, 0], self.formations[i + 1][cind, 1],
                        c=self.labels[i][rind], cmap='tab20', s=5, marker='x', label='To')
            plt.title(f"Transition {i} → {i + 1}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
