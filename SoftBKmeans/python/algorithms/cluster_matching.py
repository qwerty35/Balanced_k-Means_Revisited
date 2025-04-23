import numpy as np
from scipy.optimize import linear_sum_assignment
from balkmeans import balkmeans

class ClusterMatching:
    def __init__(self, num_clusters, clustering_params):
        self.K = num_clusters
        self.params = clustering_params

    def cluster_formations(self):
        for path in self.files:
            data, labels = self.run_clustering(path)
            centroids = self.compute_centroids(data, labels)
            self.formations.append(data)
            self.labels.append(labels)
            self.centroids.append(centroids)
            self.plot_clusters(data, labels, centroids, os.path.basename(path))

    def run_clustering(self, file_path):
        data = np.loadtxt(file_path)
        labels = balkmeans(data.tolist(), **self.params)
        return data, np.array(labels)

    def compute_centroids(self, data, labels):
        centroids = []
        for k in range(self.K):
            points = data[labels == k]
            centroids.append(np.mean(points, axis=0))
        return np.array(centroids)

    def compute_cluster_cost_matrix(self, fpos, flabels, tpos, tlabels):
        cost = np.zeros((self.K, self.K))
        for i in range(self.K):
            fi = fpos[flabels == i]
            for j in range(self.K):
                tj = tpos[tlabels == j]
                cost[i, j] = np.mean(np.linalg.norm(fi[:, None] - tj[None, :, :], axis=2))
        return cost

    def match_all(self, formations, labels):
        transitions = []
        for i in range(len(formations) - 1):
            fpos, tpos = formations[i], formations[i+1]
            flab, tlab = labels[i], labels[i+1]
            cost = self.compute_cluster_cost_matrix(fpos, flab, tpos, tlab)
            cmatch = list(zip(*linear_sum_assignment(cost)))
            rinds, cinds = [], []
            for (fc, tc) in cmatch:
                fidx = np.where(flab == fc)[0]
                tidx = np.where(tlab == tc)[0]
                assert len(fidx) == len(tidx)
                D = np.linalg.norm(fpos[fidx][:, None] - tpos[tidx][None, :, :], axis=2)
                r, c = linear_sum_assignment(D)
                rinds.extend(fidx[r])
                cinds.extend(tidx[c])
            transitions.append((np.array(rinds), np.array(cinds)))
        return transitions

    def compute_total_distances(self, formations, transitions):
        N = formations[0].shape[0]
        dists = np.zeros(N)
        current_indices = np.arange(N)

        for i, (rind, cind) in enumerate(transitions):
            moved = np.linalg.norm(formations[i][rind] - formations[i+1][cind], axis=1)
            dists += moved
            current_indices = cind
        return dists
