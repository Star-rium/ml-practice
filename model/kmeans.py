import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self):
        pass

    def init_centers(self, X, k):
        return X[np.random.choice(X.shape[0], size=k, replace=False)] # choose k random points as initial centers
    
    def assign_label(self, X, centers):
        D = cdist(X, centers)
        return np.argmin(D, axis=1)

    def update_centers(self, X, labels, k):
        centers = np.zeros((k, X.shape[1]))
        for k in range(k):
            Xk = X[labels == k, :]
            centers[k, :] = np.mean(Xk, axis=0)
        return centers

    def has_converged(self, centers, new_centers):
        return np.allclose(centers, new_centers)

    def fit(self, X, k, max_iters):
        centers = [self.init_centers(X, k)]
        labels = []
        it = 0
        while it < max_iters:
            labels.append(self.assign_label(X, centers[-1]))
            newCenters = self.update_centers(X, labels[-1], k)
            if self.has_converged(centers[-1], newCenters):
                break
            centers.append(newCenters)
            it+=1

        return (centers[-1], labels[-1], it) # return only the last iteration

if __name__ == "__main__":
    #test block
    X = np.array([[2, 5], [3, 2], [5, 1], [6, 4], [7, 3], [4, 4]])
    model = KMeans()
    res_center, label, it = model.fit(X, 2, 100)

    print(res_center, label, it)
