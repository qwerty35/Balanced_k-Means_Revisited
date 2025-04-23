import numpy as np
from abc import ABC, abstractmethod
from balkmeans import balkmeans
import matplotlib.pyplot as plt
import os

class ClusterMatchingBase(ABC):
    def __init__(self, formation_files, num_clusters, clustering_params, cluster_matcher,
                 agent_matcher):
        self.files = formation_files
        self.K = num_clusters
        self.params = clustering_params
        self.cluster_matcher = cluster_matcher
        self.agent_matcher = agent_matcher
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

    @abstractmethod
    def match_all(self):
        pass

    @abstractmethod
    def summarize(self):
        pass

    def visualize_transitions(self):
        for i, (rind, cind) in enumerate(self.transitions):
            self.plot_transitions(
                self.formations[i], self.formations[i + 1],
                self.labels[i], rind, cind, f"Transition {i} → {i + 1}"
            )

    def plot_clusters(self, data, labels, centroids, title):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=80)
        plt.title(f"Clustering: {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_transitions(self, from_pos, to_pos, from_labels, row_ind, col_ind, title):
        plt.figure(figsize=(10, 10))
        for i, j in zip(row_ind, col_ind):
            color = from_labels[i] % 20
            plt.plot([from_pos[i, 0], to_pos[j, 0]],
                     [from_pos[i, 1], to_pos[j, 1]],
                     color=plt.cm.tab20(color), alpha=0.2, linewidth=0.5)

        plt.scatter(from_pos[:, 0], from_pos[:, 1], c=from_labels, cmap='tab20', s=5, label="From")
        plt.scatter(to_pos[col_ind, 0], to_pos[col_ind, 1], c=from_labels[row_ind], cmap='tab20', s=5, marker='x', label="To")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()