#!/usr/bin/env python

from SoftBKmeans.python.hungarian_cluster_matching import HungarianClusterMatching
from SoftBKmeans.python.lbap_cluster_matching import LBAPClusterMatching
from SoftBKmeans.python.matching_functions import mean_cost_matrix, hungarian_match, bottleneck_cost_matrix, bottleneck_match

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

hungarian_cluster_matching = HungarianClusterMatching(
    formation_files,
    num_clusters=num_clusters,
    clustering_params=bkm_params,
    cluster_matcher=mean_cost_matrix,
    agent_matcher=hungarian_match
)
hungarian_cluster_matching.run()

# lbap_cluster_matching = LBAPClusterMatching(
#     formation_files,
#     num_clusters=num_clusters,
#     clustering_params=bkm_params,
#     cluster_matcher=bottleneck_cost_matrix,
#     agent_matcher=bottleneck_match
# )
# lbap_cluster_matching.run()