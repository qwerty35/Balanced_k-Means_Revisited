import numpy as np
import matplotlib.pyplot as plt
import os
from balkmeans import balkmeans
import matplotlib as mpl

class OneStepClusterMatching:
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
        self.centroids = []

    def run(self):
        self.cluster_formations()
        self.match_all()
        self.visualize_transitions()
        self.summarize()
        self.plot_max_movement_agent_path()

    def cluster_formations(self):
        for path in self.files:
            data = np.loadtxt(path)
            labels = balkmeans(data.tolist(), **self.params)
            centroids = self.compute_centroids(data, labels)
            self.formations.append(data)
            self.labels.append(np.array(labels))
            self.centroids.append(centroids)
            self.plot_clusters(data, labels, centroids, os.path.basename(path))

    def compute_centroids(self, data, labels):
        return np.array([
            np.mean(data[labels == k], axis=0)
            for k in range(self.K)
        ])

    def match_all(self):
        self.transitions = []
        for i in range(len(self.formations) - 1):
            fpos, tpos = self.formations[i], self.formations[i + 1]
            flab, tlab = self.labels[i], self.labels[i + 1]

            cluster_cost = self.cluster_cost_fn(self.agent_cost_fn, fpos, flab, tpos, tlab, self.K)
            row_ind, col_ind = self.cluster_assign_fn(cluster_cost)

            row_inds, col_inds = [], []
            for fc, tc in zip(row_ind, col_ind):
                fidx = np.where(flab == fc)[0]
                tidx = np.where(tlab == tc)[0]
                A, B = fpos[fidx], tpos[tidx]
                agent_cost = self.agent_cost_fn(A, B)
                r, c = self.agent_assign_fn(agent_cost)
                row_inds.extend(fidx[r])
                col_inds.extend(tidx[c])

            self.transitions.append((np.array(row_inds), np.array(col_inds)))

    def summarize(self):
        num_agents = self.formations[0].shape[0]
        total_dists = np.zeros(num_agents)
        current_indices = np.arange(num_agents)

        for i, (rind, cind) in enumerate(self.transitions):
            # 현재 매핑 정보를 기반으로 index 추적
            next_indices = np.zeros_like(current_indices)
            moved = np.zeros_like(current_indices, dtype=float)

            for idx, src in enumerate(current_indices):
                if src in rind:
                    src_idx = np.where(rind == src)[0][0]
                    dst_idx = cind[src_idx]
                    moved[idx] = np.linalg.norm(self.formations[i][src] - self.formations[i + 1][dst_idx])
                    next_indices[idx] = dst_idx
                else:
                    # unmapped → 거리 0 유지
                    next_indices[idx] = src

            total_dists += moved
            current_indices = next_indices.copy()

            print(f"[Transition {i} → {i + 1}] 평균 이동 거리: {np.mean(moved):.2f}, 최대 이동 거리: {np.max(moved):.2f}")

        print(f"\n[Total] 평균 이동 거리: {np.mean(total_dists):.2f}, 최대 이동 거리: {np.max(total_dists):.2f}")

    def plot_clusters(self, data, labels, centroids, title):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=50, alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=80)
        plt.title(f"Clustering: {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

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
        cmap = plt.cm.get_cmap('nipy_spectral', num_clusters)  # cluster 수에 맞게 색 만들기
        norm = mpl.colors.Normalize(vmin=0, vmax=num_clusters - 1)

        for i, j in zip(row_ind, col_ind):
            color_val = from_labels[i]
            plt.plot([from_pos[i, 0], to_pos[j, 0]],
                     [from_pos[i, 1], to_pos[j, 1]],
                     color=cmap(norm(color_val)), alpha=0.2, linewidth=2)

        plt.scatter(from_pos[row_ind, 0], from_pos[row_ind, 1],
                    c=from_labels[row_ind], cmap=cmap, norm=norm, s=50, label="From")

        plt.scatter(to_pos[col_ind, 0], to_pos[col_ind, 1],
                    c=from_labels[row_ind], cmap=cmap, norm=norm, s=50, marker='x', label="To")

        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_max_movement_agent_path(self):
        num_agents = self.formations[0].shape[0]
        total_dists = np.zeros(num_agents)
        current_indices = np.arange(num_agents)
        full_paths = [[idx] for idx in range(num_agents)]

        for i, (rind, cind) in enumerate(self.transitions):
            next_indices = np.zeros_like(current_indices)
            for idx, src in enumerate(current_indices):
                if src in rind:
                    src_idx = np.where(rind == src)[0][0]
                    dst_idx = cind[src_idx]
                    dist = np.linalg.norm(self.formations[i][src] - self.formations[i + 1][dst_idx])
                    total_dists[idx] += dist
                    next_indices[idx] = dst_idx
                    full_paths[idx].append(dst_idx)
                else:
                    next_indices[idx] = src
                    full_paths[idx].append(src)
            current_indices = next_indices.copy()

        max_idx = np.argmax(total_dists)
        path = full_paths[max_idx]

        # 시각화
        plt.figure(figsize=(10, 10))
        for i in range(len(path) - 1):
            pt1 = self.formations[i][path[i]]
            pt2 = self.formations[i + 1][path[i + 1]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=1)

        plt.scatter(*zip(*[self.formations[i][path[i]] for i in range(len(path))]),
                    c='blue', s=10, label='Agent Path')
        plt.title(f"Most Moving Agent Path (ID: {max_idx}, Total: {total_dists[max_idx]:.2f})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()