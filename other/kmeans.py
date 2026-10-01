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

# ==============================================================================
# 3. Matplotlib Animation Function
# ==============================================================================
def animate_kmeans(X, centers, labels, interval=900, save_path=None):
    num_frames = len(labels)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    cmap_light = ListedColormap(['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'])
    # Fixed plot boundaries
    pad = 8
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    # Meshgrid for decision boundary regions
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    def update(frame):
        ax.clear()
        current_centers = centers[frame]
        current_labels = labels[frame]
        # 1. Background Voronoi cluster regions
        grid_dist = cdist(grid_pts, current_centers)
        grid_labels = np.argmin(grid_dist, axis=1).reshape(xx.shape)
        ax.contourf(xx, yy, grid_labels, levels=np.arange(-0.5, k + 0.5),
                    cmap=cmap_light, alpha=0.3, zorder=0)
        ax.contour(xx, yy, grid_labels, levels=np.arange(k),
                   colors='grey', linewidths=0.8, linestyles='--', alpha=0.4, zorder=1)
        # 2. Scatter customer data points
        for c_id in range(k):
            pts = X[current_labels == c_id]
            ax.scatter(pts[:, 0], pts[:, 1], color=colors[c_id], s=55,
                       edgecolors='k', linewidths=0.5, zorder=3, label=f'Cluster {c_id + 1}')
        # 3. Centroid movement trajectory trails
        for c_id in range(k):
            traj = np.array([centers[step][c_id] for step in range(frame + 1)])
            if len(traj) > 1:
                ax.plot(traj[:, 0], traj[:, 1], color=colors[c_id],
                        linestyle='--', linewidth=2.0, alpha=0.85, zorder=4)
                ax.scatter(traj[:-1, 0], traj[:-1, 1], color=colors[c_id],
                           s=35, marker='o', alpha=0.6, zorder=4)
        # 4. Current centroids
        for c_id in range(k):
            c = current_centers[c_id]
            ax.scatter(c[0], c[1], color=colors[c_id], marker='X', s=240,
                       edgecolors='black', linewidths=2.2, zorder=6,
                       label=f'Centroid {c_id + 1}' if frame == 0 else '')
        # 5. Compute WCSS (Within-Cluster Sum of Squares)
        wcss = sum(np.sum((X[current_labels == j] - current_centers[j])**2) for j in range(k))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Annual Income (k$)', fontsize=12)
        ax.set_ylabel('Spending Score (1-100)', fontsize=12)
        status = 'CONVERGED!' if frame == num_frames - 1 else 'Optimizing...'
        ax.set_title(f'K-Means Clustering — Iteration {frame}/{num_frames - 1} ({status})\n'
                     f'Within-Cluster Sum of Squares (WCSS): {wcss:,.1f}',
                     fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
        ax.grid(True, linestyle=':', alpha=0.5)
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, repeat=True)
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000 / interval)
        print(f"Animation saved to: {save_path}")
    return anim
# ==============================================================================
# 4. Interactive Frame-by-Frame Controller (Slider + Prev / Play / Next)
# ==============================================================================
class KMeansFrameViewer:
    def __init__(self, X, centers, labels):
        self.X = X
        self.centers = centers
        self.labels = labels
        self.num_frames = len(labels)
        self.current_frame = 0
        self.is_playing = False
        self.timer = None
        self.k = len(centers[0])
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.cmap_light = ListedColormap(['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'])
        pad = 8
        self.x_min, self.x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
        self.y_min, self.y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
        xx, yy = np.meshgrid(np.linspace(self.x_min, self.x_max, 100), np.linspace(self.y_min, self.y_max, 100))
        self.xx, self.yy = xx, yy
        self.grid_pts = np.c_[xx.ravel(), yy.ravel()]
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.subplots_adjust(bottom=0.22)
        # Slider
        ax_slider = self.fig.add_axes([0.18, 0.11, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Iteration', 0, self.num_frames - 1, valinit=0, valstep=1, valfmt='%d')
        self.slider.on_changed(self.on_slider_change)
        # Buttons
        ax_prev = self.fig.add_axes([0.15, 0.03, 0.12, 0.045])
        self.btn_prev = Button(ax_prev, '◀ Prev')
        self.btn_prev.on_clicked(self.prev_frame)
        ax_play = self.fig.add_axes([0.30, 0.03, 0.14, 0.045])
        self.btn_play = Button(ax_play, '▶ Play')
        self.btn_play.on_clicked(self.toggle_play)
        ax_next = self.fig.add_axes([0.47, 0.03, 0.12, 0.045])
        self.btn_next = Button(ax_next, 'Next ▶')
        self.btn_next.on_clicked(self.next_frame)
        ax_reset = self.fig.add_axes([0.62, 0.03, 0.12, 0.045])
        self.btn_reset = Button(ax_reset, '↺ Reset')
        self.btn_reset.on_clicked(self.reset)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.render_frame(0)
    def render_frame(self, frame):
        self.ax.clear()
        self.current_frame = frame
        current_centers = self.centers[frame]
        current_labels = self.labels[frame]
        grid_dist = cdist(self.grid_pts, current_centers)
        grid_labels = np.argmin(grid_dist, axis=1).reshape(self.xx.shape)
        self.ax.contourf(self.xx, self.yy, grid_labels, levels=np.arange(-0.5, self.k + 0.5),
                         cmap=self.cmap_light, alpha=0.3, zorder=0)
        for c_id in range(self.k):
            pts = self.X[current_labels == c_id]
            self.ax.scatter(pts[:, 0], pts[:, 1], color=self.colors[c_id], s=55,
                            edgecolors='k', linewidths=0.5, zorder=3, label=f'Cluster {c_id + 1}')
        for c_id in range(self.k):
            traj = np.array([self.centers[step][c_id] for step in range(frame + 1)])
            if len(traj) > 1:
                self.ax.plot(traj[:, 0], traj[:, 1], color=self.colors[c_id],
                             linestyle='--', linewidth=2.0, alpha=0.85, zorder=4)
                self.ax.scatter(traj[:-1, 0], traj[:-1, 1], color=self.colors[c_id],
                                s=35, marker='o', alpha=0.6, zorder=4)
        for c_id in range(self.k):
            c = current_centers[c_id]
            self.ax.scatter(c[0], c[1], color=self.colors[c_id], marker='X', s=240,
                            edgecolors='black', linewidths=2.2, zorder=6)
        wcss = sum(np.sum((self.X[current_labels == j] - current_centers[j])**2) for j in range(self.k))
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xlabel('Annual Income (k$)', fontsize=12)
        self.ax.set_ylabel('Spending Score (1-100)', fontsize=12)
        status = 'CONVERGED!' if frame == self.num_frames - 1 else 'Optimizing...'
        self.ax.set_title(f'K-Means — Iteration {frame}/{self.num_frames - 1} ({status})\n'
                          f'WCSS: {wcss:,.1f}', fontsize=12, fontweight='bold', pad=10)
        self.ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
        self.ax.grid(True, linestyle=':', alpha=0.5)
        self.fig.canvas.draw_idle()
    def on_slider_change(self, val):
        frame = int(round(val))
        if frame != self.current_frame:
            self.render_frame(frame)
    def next_frame(self, event=None):
        if self.current_frame < self.num_frames - 1:
            self.slider.set_val(self.current_frame + 1)
    def prev_frame(self, event=None):
        if self.current_frame > 0:
            self.slider.set_val(self.current_frame - 1)
    def reset(self, event=None):
        self.stop_play()
        self.slider.set_val(0)
    def toggle_play(self, event=None):
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()
    def start_play(self):
        self.is_playing = True
        self.btn_play.label.set_text('❚❚ Pause')
        self.timer = self.fig.canvas.new_timer(interval=800)
        self.timer.add_callback(self._timer_step)
        self.timer.start()
    def stop_play(self):
        self.is_playing = False
        self.btn_play.label.set_text('▶ Play')
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
    def _timer_step(self):
        if self.current_frame < self.num_frames - 1:
            self.slider.set_val(self.current_frame + 1)
        else:
            self.stop_play()
    def on_key_press(self, event):
        if event.key in ['right', 'n']:
            self.next_frame()
        elif event.key in ['left', 'p']:
            self.prev_frame()
        elif event.key == ' ':
            self.toggle_play()
        elif event.key in ['home', 'r']:
            self.reset()
        elif event.key == 'end':
            self.slider.set_val(self.num_frames - 1)
# ==============================================================================
# 5. Launch Animation or Interactive Viewer
# ==============================================================================
# # Option A: Automatic looping animation (also saves to GIF)
# anim = animate_kmeans(X, centers, labels, interval=800, save_path='kmeans_clustering.gif')
# plt.show()
# Option B: Interactive Frame-by-Frame Player (with buttons, slider, and keyboard keys)
viewer = KMeansFrameViewer(X, centers, labels)
plt.show()