import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
from scipy.spatial.distance import cdist

df = pd.read_csv('./datasets/Mall_Customers.csv')
df.head()

features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X = features.to_numpy(dtype=np.float64)
np.random.seed(11)
k = 5

def kmeans_init_centers(X, k):
    return X[np.random.choice(X.shape[0], size=k, replace=False)]

def kmeans_assign_labels(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis=1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k, :] = np.mean(Xk, axis=0)
    return centers

def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))

def kmeans(X, k):
    centers = [kmeans_init_centers(X, k)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        newCenters = kmeans_update_centers(X, labels[-1], k)
        if has_converged(centers[-1], newCenters):
            break
        centers.append(newCenters)
        it+=1

    return(centers, labels, it)

(centers, labels, it) = kmeans(X, k)
print(f"Converged at iteration: {it}")

