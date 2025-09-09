import numpy as np
import matplotlib.pyplot as plt
import os
from balkmeans import balkmeans
import matplotlib as mpl

class GreedyClusterMatching:
    def __init__(self, formation_files, num_clusters, clustering_params,
                 cluster_cost_fn, cluster_assign_fn,
                 agent_cost_fn, agent_assign_fn, max_iter = 10):
        self.files = formation_files
        self.K = num_clusters
        self.params = clustering_params

        self.cluster_cost_fn = cluster_cost_fn
        self.cluster_assign_fn = cluster_assign_fn
        self.agent_cost_fn = agent_cost_fn
        self.agent_assign_fn = agent_assign_fn

        self.max_iter = max_iter

        self.formations = []
        self.labels = []
        self.centroids = []

    def run(self):
        self.cluster_formations()
        self.match_all()
        self.summarize()
        self.post_optimize_layerwise()
        self.visualize_transitions()
        self.summarize()
        self.plot_max_movement_agent_path()

    def cluster_formations(self):
        for path in self.files:
            data = np.loadtxt(path)
            labels = balkmeans(data.tolist(), **self.params)
            centroids = self.compute_centroids(data, labels)
            self.formations.append(data)
            self.labels.append(np.array(labels))
            self.centroids.append(centroids)
            self.plot_clusters(data, labels, centroids, os.path.basename(path))

    def compute_centroids(self, data, labels):
        return np.array([
            np.mean(data[labels == k], axis=0)
            for k in range(self.K)
        ])

    def match_all(self):
        self.transitions = []
        accumulated_costs = np.zeros(self.K)

        for i in range(len(self.formations) - 1):
            fpos, tpos = self.formations[i], self.formations[i + 1]
            flab, tlab = self.labels[i], self.labels[i + 1]

            cluster_cost = self.cluster_cost_fn(self.agent_cost_fn, fpos, flab, tpos, tlab, self.K)
            # 누적된 이동 cost를 현재 cost matrix에 반영
            for fc in range(self.K):
                cluster_cost[fc, :] += accumulated_costs[fc]

            row_ind, col_ind = self.cluster_assign_fn(cluster_cost)

            row_inds, col_inds = [], []
            for fc, tc in zip(row_ind, col_ind):
                fidx = np.where(flab == fc)[0]
                tidx = np.where(tlab == tc)[0]
                A, B = fpos[fidx], tpos[tidx]
                agent_cost = self.agent_cost_fn(A, B)
                r, c = self.agent_assign_fn(agent_cost)
                row_inds.extend(fidx[r])
                col_inds.extend(tidx[c])

            self.transitions.append((np.array(row_inds), np.array(col_inds)))

            # 누적 cost 업데이트
            new_accumulated_costs = np.zeros(self.K)
            for fc, tc in zip(row_ind, col_ind):
                new_accumulated_costs[tc] = cluster_cost[fc, tc]
            accumulated_costs = new_accumulated_costs.copy()

    def summarize(self):
        num_agents = self.formations[0].shape[0]
        total_dists = np.zeros(num_agents)
        current_indices = np.arange(num_agents)

        for i, (rind, cind) in enumerate(self.transitions):
            # 현재 매핑 정보를 기반으로 index 추적
            next_indices = np.zeros_like(current_indices)
            moved = np.zeros_like(current_indices, dtype=float)

            for idx, src in enumerate(current_indices):
                if src in rind:
                    src_idx = np.where(rind == src)[0][0]
                    dst_idx = cind[src_idx]
                    moved[idx] = np.linalg.norm(self.formations[i][src] - self.formations[i + 1][dst_idx])
                    next_indices[idx] = dst_idx
                else:
                    # unmapped → 거리 0 유지
                    next_indices[idx] = src

            total_dists += moved
            current_indices = next_indices.copy()

            print(f"[Transition {i} → {i + 1}] 평균 이동 거리: {np.mean(moved):.2f}, 최대 이동 거리: {np.max(moved):.2f}")

        print(f"\n[Total] 평균 이동 거리: {np.mean(total_dists):.2f}, 최대 이동 거리: {np.max(total_dists):.2f}")

    def plot_clusters(self, data, labels, centroids, title):
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=50, alpha=0.6)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=80)
        plt.title(f"Clustering: {title}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_transitions(self):
        for i, (rind, cind) in enumerate(self.transitions):
            self.plot_transitions(
                self.formations[i], self.formations[i + 1],
                self.labels[i], rind, cind, f"Transition {i} → {i + 1}"
            )

    def plot_transitions(self, from_pos, to_pos, from_labels, row_ind, col_ind, title):
        plt.figure(figsize=(10, 10))

        # cluster 수에 맞게 colormap 생성
        num_clusters = self.K
        cmap = plt.cm.get_cmap('nipy_spectral', num_clusters)  # cluster 수에 맞게 색 만들기
        norm = mpl.colors.Normalize(vmin=0, vmax=num_clusters - 1)

        for i, j in zip(row_ind, col_ind):
            color_val = from_labels[i]
            plt.plot([from_pos[i, 0], to_pos[j, 0]],
                     [from_pos[i, 1], to_pos[j, 1]],
                     color=cmap(norm(color_val)), alpha=0.2, linewidth=5.0)

        plt.scatter(from_pos[row_ind, 0], from_pos[row_ind, 1],
                    c=from_labels[row_ind], cmap=cmap, norm=norm, s=50, label="From")

        plt.scatter(to_pos[col_ind, 0], to_pos[col_ind, 1],
                    c=from_labels[row_ind], cmap=cmap, norm=norm, s=50, marker='x', label="To")

        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _compute_global_totals(self, transitions=None):
        """
        모든 에이전트의 전 구간 누적 이동거리(total_dists)와
        그 최댓값(global_max)을 계산.
        """
        if transitions is None:
            transitions = self.transitions

        num_agents = self.formations[0].shape[0]
        total_dists = np.zeros(num_agents, dtype=float)
        current_indices = np.arange(num_agents)

        for t, (rind, cind) in enumerate(transitions):
            next_indices = np.empty_like(current_indices)
            moved = np.zeros_like(current_indices, dtype=float)

            # rind: 단계 t의 source index들, cind: 단계 t+1의 dest index들
            # current_indices의 각 src가 rind에 있으면 cind로 이동
            pos = {src: dst for src, dst in zip(rind.tolist(), cind.tolist())}
            for idx, src in enumerate(current_indices):
                dst = pos.get(src, src)
                moved[idx] = np.linalg.norm(
                    self.formations[t][src] - self.formations[t + 1][dst]
                )
                next_indices[idx] = dst

            total_dists += moved
            current_indices = next_indices

        global_max = float(np.max(total_dists)) if total_dists.size else 0.0
        return total_dists, global_max

    def post_optimize_layerwise(self, max_iter=5):
        """
        전역 bottleneck(개별 에이전트 누적거리의 최댓값)을 직접 줄이는 국소 탐색.
        각 단계 i에 대해 새로운 매칭을 제안하고,
        전 구간 최댓값(global_max)이 줄어드는 경우에만 적용.
        """
        improved = True
        it = 0
        _, base_global_max = self._compute_global_totals()
        print(f"[Init] Global max total distance: {base_global_max:.4f}")

        while improved and it < max_iter:
            improved = False
            it += 1
            print(f"\n[Post-Optimization Iteration {it}]")

            for i in range(len(self.formations) - 1):
                fpos, tpos = self.formations[i], self.formations[i + 1]
                flab, tlab = self.labels[i], self.labels[i + 1]

                # --- 기존 매칭 보관
                old_rind, old_cind = self.transitions[i]

                # --- 새 클러스터-클러스터 비용 (원 코드 로직 유지)
                cluster_cost = self.cluster_cost_fn(
                    self.agent_cost_fn, fpos, flab, tpos, tlab, self.K
                )

                # (선택) 간단한 미래 비용 가중(너무 크면 떨림) — 필요 없으면 제거
                if i + 1 < len(self.formations) - 1:
                    f2, f2_lab = self.formations[i + 2], self.labels[i + 2]
                    future_cost = self.cluster_cost_fn(
                        self.agent_cost_fn, tpos, tlab, f2, f2_lab, self.K
                    )
                    cluster_cost = cluster_cost + 0.5 * future_cost

                # --- 클러스터 매칭 결정 (사용 중인 cluster_assign_fn)
                row_cl, col_cl = self.cluster_assign_fn(cluster_cost)

                # --- 클러스터 내부 에이전트 매칭(bottleneck/헝가리안 등)
                cand_row_inds, cand_col_inds = [], []
                for fc, tc in zip(row_cl, col_cl):
                    fidx = np.where(flab == fc)[0]
                    tidx = np.where(tlab == tc)[0]

                    # 안전장치: 비어있거나 크기가 다르면 가능한 만큼만 매칭
                    n = min(len(fidx), len(tidx))
                    if n == 0:
                        continue
                    A = fpos[fidx[:n]]
                    B = tpos[tidx[:n]]

                    agent_cost = self.agent_cost_fn(A, B)
                    r, c = self.agent_assign_fn(agent_cost)
                    cand_row_inds.extend(fidx[:n][r])
                    cand_col_inds.extend(tidx[:n][c])

                cand_row_inds = np.array(cand_row_inds, dtype=int)
                cand_col_inds = np.array(cand_col_inds, dtype=int)

                # --- 전역 목적 평가: 후보 해로 교체한 경우의 global_max
                trial_transitions = list(self.transitions)
                trial_transitions[i] = (cand_row_inds, cand_col_inds)

                _, trial_global_max = self._compute_global_totals(trial_transitions)

                # --- 채택 여부
                if trial_global_max + 1e-9 < base_global_max:
                    print(f"  step {i}: improved global max {base_global_max:.4f} → {trial_global_max:.4f}")
                    self.transitions[i] = (cand_row_inds, cand_col_inds)
                    base_global_max = trial_global_max
                    improved = True
                else:
                    # 되돌리기 (trial_transitions는 지역변수이므로 원복 불필요)
                    print(f"  step {i}: no improvement (stay {base_global_max:.4f})")

        print(f"\n[Done] Global max total distance: {base_global_max:.4f}")

    def post_optimize_sa(
            self,
            max_iter=300,  # 총 시도 횟수
            T0=1.0,  # 초기 온도
            alpha=0.97,  # 냉각율 (0<alpha<1)
            future_weight=0.3,  # 미래 단계를 약하게 반영(0이면 미사용)
            noise_sigma=1e-3,  # 클러스터 비용에 아주 작은 가우시안 노이즈
            verbose=True
    ):
        """
        Simulated Annealing으로 전역 bottleneck(개별 에이전트 누적거리의 최댓값)을 직접 줄이는 확률적 탐색.
        - 한 번에 임의의 step i를 선택하여 그 step의 매칭을 '재생성'한 후보해를 만들고,
          global bottleneck 개선 정도(Δ)에 따라 확률적으로 수락.
        - Δ < 0 이면 무조건 수락, Δ >= 0 이면 확률 exp(-Δ / T)로 수락.
        """

        # 현재 기준 상태
        best_transitions = list(self.transitions)
        _, best_global_max = self._compute_global_totals(best_transitions)
        if verbose:
            print(f"[SA-Init] Global max total distance: {best_global_max:.6f}")

        T = float(T0)
        rng = np.random.default_rng()

        n_steps = len(self.formations) - 1
        if n_steps <= 0:
            if verbose:
                print("[SA] No steps to optimize.")
            return

        def _propose_step(i):
            """
            step i의 (row_inds, col_inds)를 '다시' 만들되,
            - cluster cost는 현 구성과 동일하게 계산
            - 약한 미래 반영(future_weight) 및 약간의 노이즈(noise_sigma)로 다양성 확보
            - 그리고 에이전트 매칭은 지정된 self.agent_assign_fn으로 재산출
            """
            fpos, tpos = self.formations[i], self.formations[i + 1]
            flab, tlab = self.labels[i], self.labels[i + 1]

            # 클러스터 비용 계산 (현재 코드 베이스와 동일 로직)
            cluster_cost = self.cluster_cost_fn(self.agent_cost_fn, fpos, flab, tpos, tlab, self.K)

            # 미래 단계의 비용을 약하게 섞어 local plateaus 탈출을 돕는다
            if future_weight > 0 and (i + 1) < n_steps:
                f2, f2_lab = self.formations[i + 2], self.labels[i + 2]
                future_cost = self.cluster_cost_fn(self.agent_cost_fn, tpos, tlab, f2, f2_lab, self.K)
                cluster_cost = cluster_cost + float(future_weight) * future_cost

            # 아주 작은 노이즈로 동률/평지 패턴에서 탈출 가능성 확보
            if noise_sigma > 0:
                cluster_cost = cluster_cost + rng.normal(0.0, noise_sigma, size=cluster_cost.shape)

            # 클러스터 매칭
            row_cl, col_cl = self.cluster_assign_fn(cluster_cost)

            # 클러스터 내부 에이전트 매칭
            cand_row_inds, cand_col_inds = [], []
            for fc, tc in zip(row_cl, col_cl):
                fidx = np.where(flab == fc)[0]
                tidx = np.where(tlab == tc)[0]
                if len(fidx) == 0 or len(tidx) == 0:
                    continue
                # 서로 수가 다르면 가능한 만큼만 매칭
                n = min(len(fidx), len(tidx))
                A = fpos[fidx[:n]]
                B = tpos[tidx[:n]]
                agent_cost = self.agent_cost_fn(A, B)
                r, c = self.agent_assign_fn(agent_cost)
                cand_row_inds.extend(fidx[:n][r])
                cand_col_inds.extend(tidx[:n][c])

            return np.array(cand_row_inds, dtype=int), np.array(cand_col_inds, dtype=int)

        # SA 루프
        for it in range(1, max_iter + 1):
            # 임의의 step 선택 (탐색 다양성 확보)
            i = int(rng.integers(0, n_steps))

            # 후보 해 구성
            cand_r, cand_c = _propose_step(i)
            trial_transitions = list(best_transitions)
            trial_transitions[i] = (cand_r, cand_c)

            # 목적 함수(전 구간 bottleneck) 평가
            _, trial_global_max = self._compute_global_totals(trial_transitions)

            # Δ 계산 (작을수록 좋음)
            delta = float(trial_global_max - best_global_max)

            # 수락 규칙
            accept = (delta < 0.0)
            if not accept:
                # 확률적 수락
                # T가 매우 작아지더라도 underflow 방지를 위해 min/max 보호
                prob = np.exp(-max(delta, 0.0) / max(T, 1e-12))
                accept = (rng.random() < prob)

            if accept:
                best_transitions = trial_transitions
                best_global_max = trial_global_max
                if verbose:
                    tag = "improved" if delta < 0 else "accepted-worse"
                    print(f"[SA {it:04d} | T={T:.4f}] {tag}: Δ={delta:+.6f} → global_max={best_global_max:.6f}")
            else:
                if verbose and (it % 50 == 0):
                    print(f"[SA {it:04d} | T={T:.4f}] rejected: Δ={delta:+.6f} (stay {best_global_max:.6f})")

            # 냉각
            T *= alpha

        # 최종 반영
        self.transitions = best_transitions
        if verbose:
            print(f"[SA-Done] Global max total distance: {best_global_max:.6f}")


    def plot_max_movement_agent_path(self):
        num_agents = self.formations[0].shape[0]
        total_dists = np.zeros(num_agents)
        current_indices = np.arange(num_agents)
        full_paths = [[idx] for idx in range(num_agents)]

        for i, (rind, cind) in enumerate(self.transitions):
            next_indices = np.zeros_like(current_indices)
            for idx, src in enumerate(current_indices):
                if src in rind:
                    src_idx = np.where(rind == src)[0][0]
                    dst_idx = cind[src_idx]
                    dist = np.linalg.norm(self.formations[i][src] - self.formations[i + 1][dst_idx])
                    total_dists[idx] += dist
                    next_indices[idx] = dst_idx
                    full_paths[idx].append(dst_idx)
                else:
                    next_indices[idx] = src
                    full_paths[idx].append(src)
            current_indices = next_indices.copy()

        max_idx = np.argmax(total_dists)
        path = full_paths[max_idx]

        # 시각화
        plt.figure(figsize=(10, 10))
        for i in range(len(path) - 1):
            pt1 = self.formations[i][path[i]]
            pt2 = self.formations[i + 1][path[i + 1]]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=1)

        plt.scatter(*zip(*[self.formations[i][path[i]] for i in range(len(path))]),
                    c='blue', s=10, label='Agent Path')
        plt.title(f"Most Moving Agent Path (ID: {max_idx}, Total: {total_dists[max_idx]:.2f})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()