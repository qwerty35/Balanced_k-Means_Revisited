#!/usr/bin/env python

from SoftBKmeans.python.cluster_matching import ClusterMatching
from SoftBKmeans.python.multi_step_cluster_matching import MultiStepClusterMatching
import matching_functions as mf

# 포메이션 데이터 파일 리스트
formation_files = [
    '../datasets/s1_generated_Ashape.txt',
    '../datasets/s1_generated_Bshape.txt',
    '../datasets/s1_generated_Cshape.txt',
    '../datasets/s1_generated_Dshape.txt'
]

# 클러스터 수 및 BKM+ 파라미터
num_clusters = 20

# BKM+ 파라미터
bkm_params = {
    'num_clusters': num_clusters,
    'maxdiff': 1,
    'partly_remaining_factor': 0.15,
    'increasing_penalty_factor': 1.01,
    'seed': 3393,
    'postprocess_iterations': 10
}

# Hungarian
cluster_matching = ClusterMatching(
    formation_files,
    num_clusters=num_clusters,
    clustering_params=bkm_params,
    cluster_cost_fn=mf.cluster_mean_cost,
    cluster_assign_fn=mf.hungarian_assign,
    agent_cost_fn=mf.euclidean_distance_cost,
    agent_assign_fn=mf.hungarian_assign
)

# LBAP
# cluster_matching = ClusterMatching(
#     formation_files,
#     num_clusters=num_clusters,
#     clustering_params=bkm_params,
#     cluster_cost_fn=mf.cluster_max_cost,
#     cluster_assign_fn=mf.bottleneck_assign,
#     agent_cost_fn=mf.euclidean_distance_cost,
#     agent_assign_fn=mf.bottleneck_assign
# )

cluster_matching.run()

# multi_cluster_matching = MultiStepClusterMatching(
#     formation_files=formation_files,
#     num_clusters=num_clusters,
#     clustering_params=bkm_params,
#     cluster_cost_fn=mf.cluster_max_cost,
#     cluster_assign_fn=mf.bottleneck_assign,
#     agent_cost_fn=mf.euclidean_distance_cost,
#     agent_assign_fn=mf.bottleneck_assign
# )
#
# multi_cluster_matching.run()