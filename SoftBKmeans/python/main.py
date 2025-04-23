#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from balkmeans import balkmeans
from scipy.optimize import linear_sum_assignment
import os

# ===== Function Definitions =====
def compute_cluster_cost_matrix(from_pos, from_labels, to_pos, to_labels, num_clusters):
    cost_matrix = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        from_indices = np.where(from_labels == i)[0]
        fpos = from_pos[from_indices]
        for j in range(num_clusters):
            to_indices = np.where(to_labels == j)[0]
            tpos = to_pos[to_indices]
            cost = np.mean(np.linalg.norm(fpos[:, None, :] - tpos[None, :, :], axis=2))
            cost_matrix[i, j] = cost
    return cost_matrix

def match_cluster_ids(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))

def match_within_matched_clusters(from_pos, from_labels, to_pos, to_labels, matched_clusters):
    total_row_inds = []
    total_col_inds = []
    for f_cluster, t_cluster in matched_clusters:
        from_indices = np.where(from_labels == f_cluster)[0]
        to_indices = np.where(to_labels == t_cluster)[0]
        assert len(from_indices) == len(to_indices), f"Cluster {f_cluster}->{t_cluster} size mismatch"
        fpos = from_pos[from_indices]
        tpos = to_pos[to_indices]
        cost_matrix = np.linalg.norm(fpos[:, None, :] - tpos[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_row_inds.extend(from_indices[row_ind])
        total_col_inds.extend(to_indices[col_ind])
    return np.array(total_row_inds), np.array(total_col_inds), dict(matched_clusters)

def visualize_transition(from_pos, to_pos, row_ind, col_ind, from_labels, to_labels, matched_dict, title):
    plt.figure(figsize=(10, 10))
    colors = plt.get_cmap('tab20').colors

    distances = []
    for i, j in zip(row_ind, col_ind):
        cluster_id = from_labels[i]
        color = colors[cluster_id % len(colors)]
        plt.plot([from_pos[i, 0], to_pos[j, 0]],
                 [from_pos[i, 1], to_pos[j, 1]],
                 color=color, linewidth=0.5, alpha=0.2)
        distances.append(np.linalg.norm(from_pos[i] - to_pos[j]))

    for cid in np.unique(from_labels):
        indices = np.where(from_labels == cid)[0]
        color = colors[cid % len(colors)]
        plt.scatter(from_pos[indices, 0], from_pos[indices, 1], c=[color], s=5, label=f'From Cluster {cid}')

    for fcid, tcid in matched_dict.items():
        indices = np.where(to_labels == tcid)[0]
        color = colors[fcid % len(colors)]
        plt.scatter(to_pos[indices, 0], to_pos[indices, 1], c=[color], s=5, marker='x', label=f'To Cluster {tcid}')

    plt.legend()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return distances

# ===== Main Execution =====
formation_files = [
    '../datasets/s1_generated_Ashape.txt',
    '../datasets/s1_generated_Bshape.txt',
    '../datasets/s1_generated_Cshape.txt'
]

num_clusters = 20
params = {
    'num_clusters': num_clusters,
    'maxdiff': 1,
    'partly_remaining_factor': 0.15,
    'increasing_penalty_factor': 1.01,
    'seed': 3393,
    'postprocess_iterations': 10
}

formations = []
labels_list = []
agent_total_distances = None

# Clustering and Centroid Visualization
for file_path in formation_files:
    print(f"\n===== Processing {os.path.basename(file_path)} =====")
    data = np.loadtxt(file_path)
    labels = balkmeans(data.tolist(), **params)
    labels_list.append(np.array(labels))
    formations.append(data)

    unique_labels, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique_labels, counts):
        print(f"  클러스터 {cluster_id}: {count}개")

    centroids = np.array([np.mean(data[np.array(labels) == cid], axis=0) for cid in range(num_clusters)])

    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=5, alpha=0.6, label='Data points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.title(f"Clustering Result of {os.path.basename(file_path)}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Transition with cluster-level Hungarian matching
for i in range(len(formations) - 1):
    from_pos = formations[i]
    to_pos = formations[i + 1]
    from_labels = labels_list[i]
    to_labels = labels_list[i + 1]

    cost_matrix = compute_cluster_cost_matrix(from_pos, from_labels, to_pos, to_labels, num_clusters)
    matched_clusters = match_cluster_ids(cost_matrix)
    print(f"\n클러스터 매칭 ({i}→{i+1}): {matched_clusters}")

    row_ind, col_ind, matched_dict = match_within_matched_clusters(from_pos, from_labels, to_pos, to_labels, matched_clusters)
    distances = np.linalg.norm(from_pos[row_ind] - to_pos[col_ind], axis=1)
    if agent_total_distances is None:
        agent_total_distances = np.zeros_like(distances)
    agent_total_distances += distances
    visualize_transition(from_pos, to_pos, row_ind, col_ind, from_labels, to_labels, matched_dict, title=f"Cluster-Level Transition {i} → {i+1}")

# Final summary across all transitions
print("\n===== 전체 에이전트 누적 이동 통계 =====")
print(f"전체 평균 누적 이동 거리: {np.mean(agent_total_distances):.2f}")
print(f"전체 최대 누적 이동 거리: {np.max(agent_total_distances):.2f}")
