import numpy as np
import matplotlib.pyplot as plt
import os
from balkmeans import balkmeans

class ClusterMatching:
    def __init__(self, formation_files, num_clusters, clustering_params,
                 cluster_match_fn, cluster_cost_fn,
                 agent_match_fn, agent_cost_fn):
        self.files = formation_files
        self.K = num_clusters
        self.params = clustering_params
        self.cluster_matcher = cluster_match_fn
        self.cost_matrix_fn = cluster_cost_fn
        self.agent_matcher = agent_match_fn
        self.agent_cost_fn = agent_cost_fn

        self.formations = []
        self.labels = []
        self.centroids = []

    def run(self):
        self.cluster_formations()
        self.match_all()
        self.visualize_transitions()
        self.summarize()

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

            cost_matrix = self.cost_matrix_fn(fpos, flab, tpos, tlab, self.K)
            row_ind, col_ind = self.cluster_matcher(cost_matrix=cost_matrix)

            row_inds, col_inds = [], []
            for fc, tc in zip(row_ind, col_ind):
                fidx = np.where(flab == fc)[0]
                tidx = np.where(tlab == tc)[0]
                A, B = fpos[fidx], tpos[tidx]
                agent_cost = self.agent_cost_fn(A, B)
                r, c = self.agent_matcher(cost_matrix=agent_cost)
                row_inds.extend(fidx[r])
                col_inds.extend(tidx[c])

            self.transitions.append((np.array(row_inds), np.array(col_inds)))

    def summarize(self):
        total_dists = np.zeros(self.formations[0].shape[0])
        for i, (rind, cind) in enumerate(self.transitions):
            moved = np.linalg.norm(self.formations[i][rind] - self.formations[i + 1][cind], axis=1)
            total_dists += moved
        print(f"\n[ClusterMatching] 평균 이동 거리: {np.mean(total_dists):.2f}")
        print(f"[ClusterMatching] 최대 이동 거리: {np.max(total_dists):.2f}")

    def plot_clusters(self, data, labels, centroids, title):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)
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
        for i, j in zip(row_ind, col_ind):
            color = from_labels[i] % 20
            plt.plot([from_pos[i, 0], to_pos[j, 0]], [from_pos[i, 1], to_pos[j, 1]],
                     color=plt.cm.tab20(color), alpha=0.2, linewidth=0.5)

        plt.scatter(from_pos[:, 0], from_pos[:, 1], c=from_labels, cmap='tab20', s=5, label="From")
        plt.scatter(to_pos[col_ind, 0], to_pos[col_ind, 1], c=from_labels[row_ind], cmap='tab20', s=5, marker='x', label="To")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
