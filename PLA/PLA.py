import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial.distance import cdist
np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)


def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):    
    return np.array_equal(h(w, X), y) 

def perceptron(X, y, w_init):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    while True:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            if h(w[-1], xi)[0] != yi: # misclassified point
                mis_points.append(mix_id[i])
                w_new = w[-1] + yi*xi 
                w.append(w_new)
                
        if has_converged(X, y, w[-1]):
            break
    return (w, mis_points)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron(X, y, w_init)


def animate_pla(X, y, weights, mis_points):
    """Show PLA updates one frame at a time with buttons or arrow keys."""
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    positive = y.ravel() == 1
    negative = y.ravel() == -1
    ax.scatter(X[1, positive], X[2, positive], color="tab:blue", label="+1")
    ax.scatter(X[1, negative], X[2, negative], color="tab:orange", label="-1")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Perceptron Learning Algorithm")
    ax.grid(alpha=0.25)
    ax.legend()

    boundary, = ax.plot([], [], color="black", label="Decision boundary")
    current_point, = ax.plot([], [], "o", markersize=13, markerfacecolor="none",
                             markeredgecolor="red", markeredgewidth=2)
    status = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    x_limits = ax.get_xlim()
    frame_index = [0]

    def update(frame):
        frame_index[0] = frame
        weight = weights[frame].ravel()
        if abs(weight[2]) > 1e-12:
            xs = np.array(x_limits)
            ys = -(weight[0] + weight[1] * xs) / weight[2]
            boundary.set_data(xs, ys)
            boundary.set_visible(True)
        elif abs(weight[1]) > 1e-12:
            x = -weight[0] / weight[1]
            boundary.set_data([x, x], ax.get_ylim())
            boundary.set_visible(True)
        else:
            boundary.set_visible(False)

        if frame == 0:
            current_point.set_data([], [])
            status.set_text(f"Frame 1/{len(weights)} — Initial weights")
        else:
            point = mis_points[frame - 1]
            current_point.set_data([X[1, point]], [X[2, point]])
            status.set_text(
                f"Frame {frame + 1}/{len(weights)} — Update {frame}: "
                f"misclassified point, class {y[0, point]:+.0f}"
            )
        status.set_bbox(dict(facecolor="white", alpha=0.8, edgecolor="none"))
        return boundary, current_point, status

    previous_ax = fig.add_axes([0.34, 0.06, 0.14, 0.07])
    next_ax = fig.add_axes([0.52, 0.06, 0.14, 0.07])
    previous_button = Button(previous_ax, "Previous")
    next_button = Button(next_ax, "Next")

    def show_frame(frame):
        update(frame)
        fig.canvas.draw_idle()

    previous_button.on_clicked(
        lambda _event: show_frame(max(0, frame_index[0] - 1))
    )
    next_button.on_clicked(
        lambda _event: show_frame(min(len(weights) - 1, frame_index[0] + 1))
    )

    def on_key(event):
        if event.key in ("right", "n"):
            show_frame(min(len(weights) - 1, frame_index[0] + 1))
        elif event.key in ("left", "p"):
            show_frame(max(0, frame_index[0] - 1))

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)
    # Keep widget references alive while the Matplotlib window is open.
    fig._pla_controls = (previous_button, next_button)
    plt.show()
animate_pla(X, y, w, m)
