#!/usr/bin/env python

from SoftBKmeans.python.cluster_matching import ClusterMatching
import matching_functions as mf

# 포메이션 데이터 파일 리스트
formation_files = [
    '../datasets/s1_generated_Ashape.txt',
    '../datasets/s1_generated_Bshape.txt',
    '../datasets/s1_generated_Cshape.txt'
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
    cluster_match_fn=mf.hungarian_assign,
    cluster_cost_fn=mf.cluster_mean_cost,
    agent_match_fn=mf.hungarian_assign,
    agent_cost_fn=mf.mean_cost
)

# LBAP
# cluster_matching = ClusterMatching(
#     formation_files,
#     num_clusters=num_clusters,
#     clustering_params=bkm_params,
#     cluster_match_fn=mf.bottleneck_assign,
#     cluster_cost_fn=mf.cluster_max_cost,
#     agent_match_fn=mf.bottleneck_assign,
#     agent_cost_fn=mf.max_cost
# )

cluster_matching.run()