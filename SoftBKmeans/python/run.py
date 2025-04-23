#!/usr/bin/env python

from algorithms.hungarian_cluster_matching import HungarianClusterMatching
from algorithms.lbap_cluster_matching import LBAPClusterMatching

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

# hungarian_cluster_matching = HungarianClusterMatching(formation_files, num_clusters=num_clusters, clustering_params=bkm_params)
# hungarian_cluster_matching.run()

lbap_cluster_matching = LBAPClusterMatching(formation_files, num_clusters=num_clusters, clustering_params=bkm_params)
lbap_cluster_matching.run()