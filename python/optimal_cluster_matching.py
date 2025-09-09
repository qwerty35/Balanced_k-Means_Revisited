import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary
import matplotlib.pyplot as plt
import matplotlib as mpl

class OptimalMultiStepClusterMatching:
    def __init__(self, formation_files, num_clusters, clustering_params,
                 cluster_cost_fn,
                 agent_cost_fn, agent_assign_fn):
        self.files = formation_files
        self.K = num_clusters
        self.params = clustering_params
        self.cluster_cost_fn = cluster_cost_fn
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
        self.plot_max_movement_agent_path()

    def cluster_formations(self):
        from balkmeans import balkmeans
        for path in self.files:
            data = np.loadtxt(path)
            labels = balkmeans(data.tolist(), **self.params)
            self.formations.append(data)
            self.labels.append(np.array(labels))

    def match_all(self):
        cost_fn = lambda f, fl, t, tl, K: self.cluster_cost_fn(self.agent_cost_fn, f, fl, t, tl, K)
        cluster_matches = self.optimal_bottleneck_matching(self.formations, self.labels, cost_fn)

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

    def optimal_hungarian_matching(self, formations, labels, cost_fn):
        T = len(formations) - 1
        K = len(np.unique(labels[0]))
        cost_matrices = []

        for t in range(T):
            fpos, flab = formations[t], labels[t]
            tpos, tlab = formations[t + 1], labels[t + 1]
            cost_matrix = cost_fn(fpos, flab, tpos, tlab, K)
            cost_matrices.append(cost_matrix)

        prob = LpProblem("OptimalHungarianMatching", LpMinimize)

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

    def optimal_bottleneck_matching(self, formations, labels, cluster_cost_fn):
        T = len(formations) - 1
        K = len(np.unique(labels[0]))

        # Precompute cost matrices
        cost_matrices = []
        for t in range(T):
            fpos, flab = formations[t], labels[t]
            tpos, tlab = formations[t + 1], labels[t + 1]
            cost_matrices.append(cluster_cost_fn(fpos, flab, tpos, tlab, K))

        prob = LpProblem("OptimalBottleneckMatching", LpMinimize)

        # Variables
        x = [[[[LpVariable(f"x_{t}_{g}_{i}_{j}", cat=LpBinary)
                for j in range(K)] for i in range(K)] for g in range(K)] for t in range(T)]

        d = [LpVariable(f"d_{g}", lowBound=0) for g in range(K)]  # group 누적 거리
        z = LpVariable("z", lowBound=0)  # 최대 누적 거리

        # Objective: minimize z
        prob += z

        # Constraints: initial position constraint
        for g in range(K):
            prob += lpSum(x[0][g][g][j] for j in range(K)) == 1

        # Constraints: single move constraint at each step
        for t in range(T):
            for g in range(K):
                prob += lpSum(x[t][g][i][j] for i in range(K) for j in range(K)) == 1

        # Constraints: one to one matching
        for t in range(T):
            for j in range(K):
                prob += lpSum(x[t][g][i][j] for g in range(K) for i in range(K)) == 1

        # Constraints: flow conservation
        for t in range(1, T):
            for g in range(K):
                for i in range(K):
                    prob += lpSum(x[t - 1][g][j][i] for j in range(K)) == lpSum(x[t][g][i][j] for j in range(K))

        # Constraints: define cumulative distance for each agent
        for g in range(K):
            total_distance_expr = lpSum(cost_matrices[t][i][j] * x[t][g][i][j]
                                        for t in range(T) for i in range(K) for j in range(K))
            prob += d[g] == total_distance_expr

        # Constraints: each d[a] ≤ z
        for g in range(K):
            prob += d[g] <= z

        # Solve
        prob.solve()

        # Extract matches
        matchings = []
        for t in range(T):
            match = []
            for i in range(K):
                for j in range(K):
                    count = sum(x[t][g][i][j].varValue for g in range(K))
                    if count > 0.5:
                        match.append((i, j))
            matchings.append(match)

        return matchings

    def summarize(self):
        total_dists = np.zeros(self.formations[0].shape[0])
        for i, (rind, cind) in enumerate(self.transitions):
            moved = np.linalg.norm(self.formations[i][rind] - self.formations[i + 1][cind], axis=1)
            print(f"[Transition {i} → {i + 1}] 평균 이동 거리: {np.mean(moved):.2f}, 최대 이동 거리: {np.max(moved):.2f}")
            total_dists += moved
        print(f"\n[Total] 평균 이동 거리: {np.mean(total_dists):.2f}, 최대 이동 거리: {np.max(total_dists):.2f}")

    # def visualize_transitions(self):
    #     for i, (rind, cind) in enumerate(self.transitions):
    #         plt.figure(figsize=(10, 10))
    #         for a, b in zip(rind, cind):
    #             plt.plot(
    #                 [self.formations[i][a, 0], self.formations[i + 1][b, 0]],
    #                 [self.formations[i][a, 1], self.formations[i + 1][b, 1]],
    #                 color=plt.cm.tab20(self.labels[i][a] % 20), alpha=0.02, linewidth=0.5
    #             )
    #         plt.scatter(self.formations[i][:, 0], self.formations[i][:, 1], c=self.labels[i][rind], cmap='tab20', s=5, label='From')
    #         plt.scatter(self.formations[i + 1][cind, 0], self.formations[i + 1][cind, 1],
    #                     c=self.labels[i][rind], cmap='tab20', s=5, marker='x', label='To')
    #         plt.title(f"Transition {i} → {i + 1}")
    #         plt.grid(True)
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()

    def visualize_transitions(self):
        for i, (rind, cind) in enumerate(self.transitions):
            self.plot_transitions(
                self.formations[i], self.formations[i + 1],
                self.labels[i], rind, cind, f"Transition {i} → {i + 1}"
            )

    def plot_transitions(self, from_pos, to_pos, from_labels, row_ind, col_ind, title):
        plt.figure(figsize=(10, 10))

        # cluster 수에 맞게 colormap 생성
        num_clusters = self.K
        cmap = plt.cm.get_cmap('hsv', num_clusters)
        norm = mpl.colors.Normalize(vmin=0, vmax=num_clusters - 1)

        for i, j in zip(row_ind, col_ind):
            color_val = from_labels[i]
            plt.plot([from_pos[i, 0], to_pos[j, 0]],
                     [from_pos[i, 1], to_pos[j, 1]],
                     color=cmap(norm(color_val)), alpha=0.02, linewidth=0.5)

        plt.scatter(from_pos[row_ind, 0], from_pos[row_ind, 1],
                    c=from_labels[row_ind], cmap=cmap, norm=norm, s=5, label="From")

        plt.scatter(to_pos[col_ind, 0], to_pos[col_ind, 1],
                    c=from_labels[row_ind], cmap=cmap, norm=norm, s=5, marker='x', label="To")

        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_max_movement_agent_path(self):
        """
        모든 에이전트의 누적 이동거리를 계산하고,
        최대로 이동한 에이전트의 시계열 경로를 2D로 시각화한다.
        """
        num_agents = self.formations[0].shape[0]
        total_dists = np.zeros(num_agents)

        # 시점별 매칭 결과(transitions)를 사용해 에이전트 인덱스 추적
        current_indices = np.arange(num_agents)  # t=0에서의 에이전트 인덱스
        full_paths = [[idx] for idx in range(num_agents)]  # 각 에이전트의 시점별 인덱스 경로

        for i, (rind, cind) in enumerate(self.transitions):
            next_indices = np.zeros_like(current_indices)
            moved_this_step = np.zeros_like(current_indices, dtype=float)

            # current_indices의 각 에이전트 src가 rind에 존재하면 cind의 대응 dst로 이동
            for idx, src in enumerate(current_indices):
                matches = np.where(rind == src)[0]
                if matches.size > 0:
                    src_idx = matches[0]
                    dst_idx = cind[src_idx]
                    dist = np.linalg.norm(self.formations[i][src] - self.formations[i + 1][dst_idx])
                    moved_this_step[idx] = dist
                    next_indices[idx] = dst_idx
                    full_paths[idx].append(dst_idx)
                else:
                    # 이 시점 매칭에 포함되지 않은 경우 자기 자신 유지(이동 0)
                    next_indices[idx] = src
                    full_paths[idx].append(src)

            total_dists += moved_this_step
            current_indices = next_indices.copy()

        # 최대로 이동한 에이전트 선택
        max_idx = int(np.argmax(total_dists))
        path = full_paths[max_idx]

        # 시각화
        plt.figure(figsize=(10, 10))
        for t in range(len(path) - 1):
            p1 = self.formations[t][path[t]]
            p2 = self.formations[t + 1][path[t + 1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=1)

        pts = [self.formations[t][path[t]] for t in range(len(path))]
        xs, ys = zip(*pts)
        plt.scatter(xs, ys, s=10, label='Agent Path')
        plt.title(f"Most Moving Agent Path (ID: {max_idx}, Total: {total_dists[max_idx]:.2f})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
