import numpy as np
import os
import matplotlib.pyplot as plt

class FormationManager:
    def __init__(self, formation_files, num_clusters, matching_strategy):
        self.files = formation_files
        self.K = num_clusters
        self.matching_strategy = matching_strategy
        self.formations = []
        self.labels = []
        self.centroids = []

    def cluster_formations(self):
        for path in self.files:
            data, labels = self.matching_strategy.run_clustering(path)
            centroids = self.matching_strategy.compute_centroids(data, labels)
            self.formations.append(data)
            self.labels.append(labels)
            self.centroids.append(centroids)
            self.plot_clusters(data, labels, centroids, os.path.basename(path))

    def transition_and_visualize(self):
        self.transitions = self.matching_strategy.match_all(self.formations, self.labels)
        for i, (rind, cind) in enumerate(self.transitions):
            self.plot_transitions(
                self.formations[i], self.formations[i+1],
                self.labels[i], self.labels[i+1],
                rind, cind, title=f"Transition {i} → {i+1}"
            )

    def summarize(self):
        dists = self.matching_strategy.compute_total_distances(self.formations, self.transitions)
        print("\n===== 전체 에이전트 누적 이동 통계 =====")
        print(f"전체 평균 누적 이동 거리: {np.mean(dists):.2f}")
        print(f"전체 최대 누적 이동 거리: {np.max(dists):.2f}")

    def plot_clusters(self, data, labels, centroids, title):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=80)
        plt.title(f"Clustering: {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_transitions(self, from_pos, to_pos, from_labels, to_labels, row_ind, col_ind, title):
        plt.figure(figsize=(10, 10))
        for i, j in zip(row_ind, col_ind):
            color = from_labels[i] % 20
            plt.plot(
                [from_pos[i, 0], to_pos[j, 0]],
                [from_pos[i, 1], to_pos[j, 1]],
                color=plt.cm.tab20(color), alpha=0.05, linewidth=0.5
            )

        # 이동 전
        plt.scatter(from_pos[:, 0], from_pos[:, 1], c=from_labels, cmap='tab20', s=5, label="From")

        # 이동 후 (색을 from_labels[row_ind]로 맞춤)
        mapped_colors = from_labels[row_ind]
        plt.scatter(to_pos[col_ind, 0], to_pos[col_ind, 1], c=mapped_colors, cmap='tab20', s=5, marker='x', label="To")

        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
