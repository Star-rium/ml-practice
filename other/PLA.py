import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis=1)

#X_hat
X = np.concatenate((np.ones((1, 2*N)), X), axis=0)

def h(w, x):
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):
    return np.array_equal(h(w, X), y)

def PLA(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    _iter = 0
    miss_points = []
    while True:
        mix = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix[i]].reshape(d, 1)
            yi = y[0, mix[i]]
            if h(w[-1], xi)[0] != yi:
                miss_points.append(int(mix[i]))
                w_new = w[-1] + yi*xi
                w.append(w_new)

        if has_converged(X, y, w[-1]):
            break

    return (w, miss_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = PLA(X, y, w_init)

# ==============================================================================
# Matplotlib Animation for Each Iteration / Step
# ==============================================================================
def animate_pla_margin(X, y, w, miss_points, interval=900, save_path=None):
    x1_coords = X[1, :]
    x2_coords = X[2, :]
    labels = y[0, :]
    pos_mask = (labels == 1)
    neg_mask = (labels == -1)
    # Fixed plot limits so the frame remains stable
    pad = 0.8
    x1_min, x1_max = x1_coords.min() - pad, x1_coords.max() + pad
    x2_min, x2_max = x2_coords.min() - pad, x2_coords.max() + pad
    x1_line = np.linspace(x1_min, x1_max, 300)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    def update(frame):
        ax.clear()
        w_curr = w[frame].ravel()
        w0, w1, w2 = w_curr
        norm_w = np.sqrt(w1**2 + w2**2)
        scores = np.dot(w[frame].T, X).ravel()
        # Check classification accuracy
        preds = np.sign(scores)
        preds[preds == 0] = 1
        misclassified = np.where(preds != labels)[0]
        is_converged = (len(misclassified) == 0)
        # 1. Scatter data points
        ax.scatter(x1_coords[pos_mask], x2_coords[pos_mask],
                   color='#1f77b4', marker='o', s=60, edgecolors='k',
                   label='Class +1', zorder=4)
        ax.scatter(x1_coords[neg_mask], x2_coords[neg_mask],
                   color='#d62728', marker='s', s=60, edgecolors='k',
                   label='Class -1', zorder=4)
        # 2. Highlight point that triggered this update (if frame > 0)
        if frame > 0 and (frame - 1) < len(miss_points):
            trig_idx = miss_points[frame - 1]
            ax.scatter(x1_coords[trig_idx], x2_coords[trig_idx],
                       s=240, facecolors='none', edgecolors='gold', linewidths=3.5,
                       label=f'Updated pt #{trig_idx} (Class {int(labels[trig_idx])})', zorder=7)
        # 3. Draw Decision Boundary & Margin Lines
        if abs(w2) > 1e-6:
            # Decision boundary: w0 + w1*x1 + w2*x2 = 0
            x2_db = -(w1 * x1_line + w0) / w2
            ax.plot(x1_line, x2_db, color='black', linewidth=2.2,
                    label=r'Decision boundary ($w^T x = 0$)', zorder=3)
            # Nearest points of each class
            pos_scores = scores[pos_mask]
            neg_scores = scores[neg_mask]
            c_pos = np.min(pos_scores)
            c_neg = np.max(neg_scores)
            pos_indices = np.where(pos_mask)[0]
            neg_indices = np.where(neg_mask)[0]
            closest_pos_idx = pos_indices[np.argmin(pos_scores)]
            closest_neg_idx = neg_indices[np.argmax(neg_scores)]
            # Positive and negative margin lines
            x2_pos = -(w1 * x1_line + w0 - c_pos) / w2
            x2_neg = -(w1 * x1_line + w0 - c_neg) / w2
            dist_pos = c_pos / norm_w
            dist_neg = -c_neg / norm_w
            ax.plot(x1_line, x2_pos, color='#1f77b4', linestyle='--', linewidth=1.5,
                    label=f'Positive margin (dist={dist_pos:.2f})', zorder=2)
            ax.plot(x1_line, x2_neg, color='#d62728', linestyle='--', linewidth=1.5,
                    label=f'Negative margin (dist={dist_neg:.2f})', zorder=2)
            # Highlight closest points defining the margin
            ax.scatter(x1_coords[closest_pos_idx], x2_coords[closest_pos_idx],
                       s=160, facecolors='none', edgecolors='#1f77b4', linewidths=2.0,
                       linestyle=':', label='Closest (+1)', zorder=5)
            ax.scatter(x1_coords[closest_neg_idx], x2_coords[closest_neg_idx],
                       s=160, facecolors='none', edgecolors='#d62728', linewidths=2.0,
                       linestyle=':', label='Closest (-1)', zorder=5)
            # Highlight margin gap when fully separated
            if is_converged:
                margin_w = (c_pos - c_neg) / norm_w
                ax.fill_between(x1_line, x2_pos, x2_neg, color='gray', alpha=0.2,
                                label=f'Margin region (width={margin_w:.3f})', zorder=1)
        else:
            x1_db = -w0 / w1
            ax.axvline(x1_db, color='black', linewidth=2.2, label=r'Decision boundary ($w^T x = 0$)')
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xlabel('$x_1$', fontsize=11)
        ax.set_ylabel('$x_2$', fontsize=11)
        status_text = 'CONVERGED!' if is_converged else f'Misclassified: {len(misclassified)}/{len(labels)}'
        ax.set_title(f'PLA Step {frame}/{len(w)-1} — {status_text}', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.6)
    anim = FuncAnimation(fig, update, frames=len(w), interval=interval, repeat=True)
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000/interval)
        print(f"Animation saved to: {save_path}")
    return anim
# Run the animation window
anim = animate_pla_margin(X, y, w, m, interval=2000, save_path='pla_margin_animation.gif')
plt.show()