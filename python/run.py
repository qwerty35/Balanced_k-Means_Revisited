#!/usr/bin/env python

from python.one_step_cluster_matching import OneStepClusterMatching
from python.greedy_cluster_matching import GreedyClusterMatching
from python.optimal_cluster_matching import OptimalMultiStepClusterMatching
from utils_export import export_results
import matching_functions as mf
import time


# --- 선택할 알고리즘 ---
# Greedy, Hungarian, LBAP, Optimal 중 하나
algo_name = "Greedy"

# 포메이션 데이터 파일 리스트
# s1: 10000 agents
# formation_files = [
#     '../datasets/s1_generated_Ashape.txt',
#     '../datasets/s1_generated_Bshape.txt',
#     '../datasets/s1_generated_Cshape.txt',
#     '../datasets/s1_generated_Dshape.txt'
# ]
# s2: 10000 agents
# formation_files = [
#     '../datasets/s2_generated_Ashape.txt',
#     '../datasets/s2_generated_Bshape.txt',
#     '../datasets/s2_generated_Cshape.txt',
#     '../datasets/s2_generated_Dshape.txt',
#     '../datasets/s2_generated_Eshape.txt',
#     '../datasets/s2_generated_Fshape.txt',
#     '../datasets/s2_generated_Gshape.txt',
#     '../datasets/s2_generated_Hshape.txt',
#     '../datasets/s2_generated_Ishape.txt',
#     '../datasets/s2_generated_Jshape.txt'
# ]
# s3: 30 agents
formation_files = [
    '../datasets/s3_generated_Ashape.txt',
    '../datasets/s3_generated_Bshape.txt',
    '../datasets/s3_generated_Cshape.txt',
    '../datasets/s3_generated_Dshape.txt',
    '../datasets/s3_generated_Eshape.txt',
    '../datasets/s3_generated_Fshape.txt',
    '../datasets/s3_generated_Gshape.txt',
    '../datasets/s3_generated_Hshape.txt',
    '../datasets/s3_generated_Ishape.txt',
    '../datasets/s3_generated_Jshape.txt'
]

# 클러스터 수
num_clusters = 10

# BKM+ 파라미터
bkm_params = {
    'num_clusters': num_clusters,
    'maxdiff': 1,
    'partly_remaining_factor': 0.15,
    'increasing_penalty_factor': 1.01,
    'seed': 3393,
    'postprocess_iterations': 10
}

# 시간 측정 시작
start_time = time.time()

# Hungarian (one-step matching)
if algo_name == "Hungarian":
    obj = OneStepClusterMatching(
        formation_files,
        num_clusters=num_clusters,
        clustering_params=bkm_params,
        cluster_cost_fn=mf.cluster_mean_cost,
        cluster_assign_fn=mf.hungarian_assign,
        agent_cost_fn=mf.euclidean_distance_cost,
        agent_assign_fn=mf.hungarian_assign
    )
    obj.run()

# LBAP (one-step matching)
elif algo_name == "LBAP":
    obj = OneStepClusterMatching(
        formation_files,
        num_clusters=num_clusters,
        clustering_params=bkm_params,
        cluster_cost_fn=mf.cluster_max_cost,
        cluster_assign_fn=mf.bottleneck_assign,
        agent_cost_fn=mf.euclidean_distance_cost,
        agent_assign_fn=mf.hungarian_assign
    )
    obj.run()

#GreedyClusterMatching (multi-step matching)
elif algo_name == "Greedy":
    obj = GreedyClusterMatching(
        formation_files=formation_files,
        num_clusters=num_clusters,
        clustering_params=bkm_params,
        cluster_cost_fn=mf.cluster_max_cost,
        cluster_assign_fn=mf.bottleneck_assign,
        agent_cost_fn=mf.euclidean_distance_cost,
        agent_assign_fn=mf.hungarian_assign
    )
    obj.run()

#OptimalClusterMatching (multi-step MILP-based optimization)
elif algo_name == "Optimal":
    obj = OptimalMultiStepClusterMatching(
        formation_files=formation_files,
        num_clusters=num_clusters,
        clustering_params=bkm_params,
        cluster_cost_fn=mf.cluster_max_cost,
        agent_cost_fn=mf.euclidean_distance_cost,
        agent_assign_fn=mf.hungarian_assign
    )
    obj.run()

else:
    raise ValueError(f"Unknown algo_name: {algo_name}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

export_results(
    obj,
    algo_name,
    formation_files,
    log_dir="../logs",
    elapsed_time=elapsed_time
)