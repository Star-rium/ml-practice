import numpy as np

class DBSCANScratch:
    def __init__(self, epsilon=5, min_samples=5):
        self.eps = epsilon
        self.min_samples = min_samples
        self.labels_ = None
        self.is_core_ = None
        self.is_border_ = None
        self.is_noise_ = None
        self.dist_matrix_ = None

    def compute_distance_matrix(self, X):
        X = np.asarray(X)
        # case where data is 1-dimensional
        if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
            x_1d = X.ravel() # flatten to 1D array

            return np.abs(x_1d[:, np.newaxis] - x_1d[np.newaxis, :])
        else:
            return np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)

    
    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]

        labels = np.zeros(n_samples, dtype=int)
        is_core = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        self.dist_matrix_ = self.compute_distance_matrix(X)

        for i in range(n_samples):
            if labels[i] != 0: continue
            neighbours = np.where(self.dist_matrix_[i] <= self.eps)[0]

            if len(neighbours) < self.min_samples:
                labels[i] = -1 
            else:
                cluster_id += 1
                labels[i] = cluster_id
                is_core[i] = True

                queue = list(neighbours)
                idx = 0
                while idx < len(queue):
                    q = queue[idx]
                    idx += 1

                    if labels[q] == -1:
                        labels[q] = cluster_id

                    elif labels[q] == 0:
                        labels[q] = cluster_id
                        q_neighbours = np.where(self.dist_matrix_[q] <= self.eps)[0]

                        if len(q_neighbours) >= self.min_samples:
                            is_core[q] = True
                            for n_idx in q_neighbours:
                                if labels[n_idx] in (0, -1) and n_idx not in queue:
                                    queue.append(n_idx)

        self.labels_ = np.where(labels > 0, labels - 1, -1)
        self.is_core_ = is_core
        self.is_noise_ = (self.labels_ == -1)
        self.is_border_ = (~self.is_core_) & (~self.is_noise_)

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def get_summary(self, X):
        X = np.asarray(X)
        clusters = {}
        for cid in sorted(set(self.labels_)):
            if cid != -1:
                clusters[f"Cluster {cid + 1}"] = X[self.labels_ == cid].tolist()

        return {
            "core_points": X[self.is_core_].tolist(),
            "border_points": X[self.is_border_].tolist(),
            "noise_points": X[self.is_noise_].tolist(),
            "clusters": clusters
        }

