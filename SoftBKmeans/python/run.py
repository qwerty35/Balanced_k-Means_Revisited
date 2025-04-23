#!/usr/bin/env python

from formation_manager import FormationManager
from algorithms.cluster_matching import ClusterMatching

# 포메이션 데이터 파일 리스트
formation_files = [
    '../datasets/s1_generated_Ashape.txt',
    '../datasets/s1_generated_Bshape.txt',
    '../datasets/s1_generated_Cshape.txt'
]

# 클러스터 수 및 BKM+ 파라미터
num_clusters = 40

# BKM+ 파라미터
bkm_params = {
    'num_clusters': num_clusters,
    'maxdiff': 1,
    'partly_remaining_factor': 0.15,
    'increasing_penalty_factor': 1.01,
    'seed': 3393,
    'postprocess_iterations': 10
}

# 클러스터 기반 매칭 전략 객체 생성
matching_strategy = ClusterMatching(num_clusters=num_clusters, clustering_params=bkm_params)

# FormationManager 초기화 및 실행
manager = FormationManager(formation_files, num_clusters, matching_strategy)
manager.cluster_formations()
manager.transition_and_visualize()
manager.summarize()
