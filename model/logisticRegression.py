import numpy as np

class LogisticRegressionScratch:
    def __init__(self, eta, max_iter, tol = 1e-4):
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y, w):
        z = self.sigmoid(np.dot(w.T, X))
        loss = -np.mean(y * np.log(z + 1e-15) + (1 - y) * np.log(1 - z + 1e-15)) # add small term to avoid log(0)
        return loss

    def fit(self, X, y):
        d, N = X.shape
        np.random.seed(31)
        self.w = [np.random.rand(d, 1)]
        self.loss_history = []

        for _ in range(self.max_iter):
            mix_id = np.random.permutation(N) # randomly shuffle the data

            for i in mix_id:
                xi = X[:, i].reshape(d, 1) # get ith data point
                yi = y[i]
                zi = self.sigmoid(np.dot(self.w[-1].T, xi))
                w_new = self.w[-1] + self.eta*(yi - zi) * xi
                self.w.append(w_new)
            
            self.loss_history.append(self.compute_loss(X, y, self.w[-1]))
            if np.linalg.norm(w_new - self.w[-1]) < self.tol:
                return self.w[-1]
        return self.w[-1]
