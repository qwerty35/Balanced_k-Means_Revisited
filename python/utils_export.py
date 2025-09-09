import csv
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import os

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def compute_index_paths(formations, transitions):
    """
    formations: list[np.ndarray], 길이 T
    transitions: list[(rind, cind)], 길이 T-1
    return: paths (N,T)  — t0 기준 agent a의 전 시점 인덱스 궤적
    """
    T = len(formations)
    N = formations[0].shape[0]
    paths = np.zeros((N, T), dtype=int)
    current = np.arange(N, dtype=int)
    paths[:, 0] = current
    for t, (rind, cind) in enumerate(transitions):
        pos = {int(s): int(d) for s, d in zip(rind.tolist(), cind.tolist())}
        current = np.array([pos.get(int(src), int(src)) for src in current], dtype=int)
        paths[:, t + 1] = current
    return paths

def write_paths_csv(csv_path, paths, formation_files=None):
    ensure_dir(csv_path)
    N, T = paths.shape
    names = [f"formation{t+1}" for t in range(T)]
    header = ["agent_id"] + [f"{nm}_idx" for nm in names]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for aid in range(N):
            w.writerow([aid] + [int(x) for x in paths[aid, :].tolist()])

def compute_total_dists(formations, paths):
    """
    paths: (N,T)
    return: total_dists (N,), max_total, mean_total
    """
    N, T = paths.shape
    totals = np.zeros(N, dtype=float)
    for t in range(T - 1):
        a = formations[t][paths[:, t]]
        b = formations[t + 1][paths[:, t + 1]]
        totals += np.linalg.norm(a - b, axis=1)
    return totals, float(totals.max()), float(totals.mean())

def save_yaml(yaml_path, data: dict):
    ensure_dir(yaml_path)
    out = {"timestamp": datetime.now().isoformat(timespec="seconds"), **data}
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

def export_results(obj, algo_name, formation_files, log_dir, elapsed_time=None):
    """
    obj: GreedyClusterMatching, OptimalMultiStepClusterMatching 등
    algo_name: "GreedyClusterMatching" | "OptimalMultiStepClusterMatching" | "OneStepClusterMatching"
    formation_files: run.py에서 사용한 formation_files 리스트
    log_dir: 로그 저장 디렉토리 (예: "logs/")
    elapsed_time: 실행 소요 시간 (초)
    """
    ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = Path(log_dir) / f"transition_{ts}.csv"
    yaml_path = Path(log_dir) / f"summary_{ts}.yaml"

    # --- CSV 저장 ---
    paths = compute_index_paths(obj.formations, obj.transitions)
    write_paths_csv(csv_path, paths, formation_files=formation_files)

    # --- YAML 저장 ---
    total_dists, max_total, mean_total = compute_total_dists(obj.formations, paths)
    meta = {
        "algorithm": algo_name,
        "params": {
            "num_clusters": getattr(obj, "K", None),
            "clustering_params": getattr(obj, "params", None),
            "agent_assign": getattr(obj, "agent_assign_fn", None).__name__ if hasattr(obj, "agent_assign_fn") else None,
            "cluster_assign": getattr(obj, "cluster_assign_fn", None).__name__ if hasattr(obj, "cluster_assign_fn") else None,
            "cluster_cost_fn": getattr(obj, "cluster_cost_fn", None).__name__ if hasattr(obj, "cluster_cost_fn") else None,
            "agent_cost_fn": getattr(obj, "agent_cost_fn", None).__name__ if hasattr(obj, "agent_cost_fn") else None,
        },
        "metrics": {
            "max_total_distance": max_total,
            "mean_total_distance": mean_total,
        },
        "agent_total_distances": [float(x) for x in total_dists.tolist()],
        "num_formations": len(obj.formations),
        "num_agents": int(obj.formations[0].shape[0]),
        "formation_files": formation_files,
    }
    if elapsed_time is not None:
        meta["elapsed_time_sec"] = float(elapsed_time)

    save_yaml(yaml_path, meta)

    print(f"[Exported] CSV: {csv_path}, YAML: {yaml_path}")