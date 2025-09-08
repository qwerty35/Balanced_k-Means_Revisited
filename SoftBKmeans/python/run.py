#!/usr/bin/env python

from SoftBKmeans.python.one_step_cluster_matching import OneStepClusterMatching
from SoftBKmeans.python.greedy_cluster_matching import GreedyClusterMatching
from SoftBKmeans.python.optimal_cluster_matching import OptimalMultiStepClusterMatching
import matching_functions as mf
import time

# 시간 측정 시작
start_time = time.time()

# 포메이션 데이터 파일 리스트
# s1: 10000 agents
# formation_files = [
#     '../datasets/s1_generated_Ashape.txt',
#     '../datasets/s1_generated_Bshape.txt',
#     '../datasets/s1_generated_Cshape.txt',
#     '../datasets/s1_generated_Dshape.txt'
# ]
# s2: 10000 agents
formation_files = [
    '../datasets/s2_generated_Ashape.txt',
    '../datasets/s2_generated_Bshape.txt',
    '../datasets/s2_generated_Cshape.txt',
    '../datasets/s2_generated_Dshape.txt',
    '../datasets/s2_generated_Eshape.txt',
    '../datasets/s2_generated_Fshape.txt',
    '../datasets/s2_generated_Gshape.txt',
    '../datasets/s2_generated_Hshape.txt',
    '../datasets/s2_generated_Ishape.txt',
    '../datasets/s2_generated_Jshape.txt'
]
# s3: 10 agents
# formation_files = [
#     '../datasets/s3_generated_Ashape.txt',
#     '../datasets/s3_generated_Bshape.txt',
#     '../datasets/s3_generated_Cshape.txt',
#     '../datasets/s3_generated_Dshape.txt',
#     '../datasets/s3_generated_Eshape.txt',
#     '../datasets/s3_generated_Fshape.txt',
#     '../datasets/s3_generated_Gshape.txt',
#     '../datasets/s3_generated_Hshape.txt',
#     '../datasets/s3_generated_Ishape.txt',
#     '../datasets/s3_generated_Jshape.txt'
# ]

# 클러스터 수
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

# Hungarian (one-step matching)
# one_step_cluster_matching = OneStepClusterMatching(
#     formation_files,
#     num_clusters=num_clusters,
#     clustering_params=bkm_params,
#     cluster_cost_fn=mf.cluster_mean_cost,
#     cluster_assign_fn=mf.hungarian_assign,
#     agent_cost_fn=mf.euclidean_distance_cost,
#     agent_assign_fn=mf.hungarian_assign
# )
# one_step_cluster_matching.run()
# one_step_cluster_matching.plot_max_movement_agent_path()

# LBAP (one-step matching)
# one_step_cluster_matching = OneStepClusterMatching(
#     formation_files,
#     num_clusters=num_clusters,
#     clustering_params=bkm_params,
#     cluster_cost_fn=mf.cluster_max_cost,
#     cluster_assign_fn=mf.bottleneck_assign,
#     agent_cost_fn=mf.euclidean_distance_cost,
#     agent_assign_fn=mf.bottleneck_assign
# )
# one_step_cluster_matching.run()

#GreedyClusterMatching (multi-step matching)
greedy_cluster_matching = GreedyClusterMatching(
    formation_files=formation_files,
    num_clusters=num_clusters,
    clustering_params=bkm_params,
    cluster_cost_fn=mf.cluster_max_cost,
    cluster_assign_fn=mf.bottleneck_assign,
    agent_cost_fn=mf.euclidean_distance_cost,
    agent_assign_fn=mf.bottleneck_assign
)
greedy_cluster_matching.run()
greedy_cluster_matching.plot_max_movement_agent_path()

#OptimalClusterMatching (multi-step MILP-based optimization)
# optimal_cluster_matching = OptimalMultiStepClusterMatching(
#     formation_files=formation_files,
#     num_clusters=num_clusters,
#     clustering_params=bkm_params,
#     cluster_cost_fn=mf.cluster_max_cost,
#     agent_cost_fn=mf.euclidean_distance_cost,
#     agent_assign_fn=mf.bottleneck_assign
# )
# optimal_cluster_matching.run()
# optimal_cluster_matching.plot_max_movement_agent_path()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")