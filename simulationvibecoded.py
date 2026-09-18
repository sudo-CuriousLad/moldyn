import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np

# --- Simulation Defaults ---
dt = 0.05
r_n = 40.0
k = 1.0
m = 1.0
bounds = 150.0
sub_steps = 4

initial_n = 20
initial_v_scale = 40.0

# Pre-allocate array: [x, y, z, vx, vy, vz]
atoms = np.zeros((initial_n, 6))


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    norms = np.linalg.norm(vec, axis=0)
    norms[norms == 0] = 1.0
    vec /= norms
    return vec


def reset_particles(n_count, v_scale):
    global atoms, time_history, ke_history
    atoms = np.zeros((n_count, 6))
    atoms[:, :3] = np.random.uniform(-bounds * 0.5, bounds * 0.5, size=(n_count, 3))
    atoms[:, 3:6] = np.transpose(sample_spherical(n_count)) * v_scale
    time_history = []
    ke_history = []


reset_particles(initial_n, initial_v_scale)


def compute_forces(pos, k_val):
    if len(pos) < 2:
        return np.zeros_like(pos)
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)

    mask = dist < r_n
    f_mag = np.zeros_like(dist)
    f_mag[mask] = k_val * (dist[mask] - 20.0) / dist[mask]
    forces = np.sum(diff * f_mag[:, :, np.newaxis], axis=1)
    return forces


def step(k_val):
    global atoms
    for j in range(3):
        hit_upper = (atoms[:, j] + atoms[:, j + 3] * dt) > bounds
        hit_lower = (atoms[:, j] + atoms[:, j + 3] * dt) < -bounds
        atoms[hit_upper | hit_lower, j + 3] *= -1

    atoms[:, :3] += atoms[:, 3:6] * dt
    forces = compute_forces(atoms[:, :3], k_val)
    atoms[:, 3:6] += (forces / m) * dt


# --- Plot & Figure Layout ---
fig = plt.figure(figsize=(10, 5.5))
plt.subplots_adjust(
    left=0.08, right=0.92, top=0.92, bottom=0.25, wspace=0.40
)

ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
ax_ke = fig.add_subplot(1, 2, 2)

ax_ke.set_title("Kinetic Energy", fontsize=10)
ax_ke.set_xlabel("Time (s)", fontsize=9)
ax_ke.set_ylabel("Total KE", fontsize=9)
ax_ke.grid(True, linestyle="--", alpha=0.5)

ke_line, = ax_ke.plot([], [], color="navy", lw=1.5)
time_history = []
ke_history = []

# --- UI Controls Layout ---
btn_ax = plt.axes([0.08, 0.12, 0.12, 0.05])
menu_btn = Button(btn_ax, "Hide Menu")

slider_axes = [
    plt.axes([0.30, 0.14, 0.58, 0.025]),  # N particles
    plt.axes([0.30, 0.09, 0.58, 0.025]),  # Velocity scale
    plt.axes([0.30, 0.04, 0.58, 0.025]),  # Force constant k
]

slider_n = Slider(
    slider_axes[0], "N Particles", 2, 80, valinit=initial_n, valstep=1
)
slider_v = Slider(
    slider_axes[1], "Velocity Scale", 5.0, 100.0, valinit=initial_v_scale
)
slider_k = Slider(
    slider_axes[2], "Force Const (k)", 0.0, 5.0, valinit=k
)


def on_n_change(val):
    reset_particles(int(val), slider_v.val)


def on_v_change(val):
    current_speeds = np.linalg.norm(atoms[:, 3:6], axis=1, keepdims=True)
    current_speeds[current_speeds == 0] = 1.0
    atoms[:, 3:6] = (atoms[:, 3:6] / current_speeds) * val


menu_visible = True


def toggle_menu(event):
    global menu_visible
    menu_visible = not menu_visible
    for s_ax in slider_axes:
        s_ax.set_visible(menu_visible)
    menu_btn.label.set_text("Hide Menu" if menu_visible else "Show Menu")
    fig.canvas.draw_idle()


slider_n.on_changed(on_n_change)
slider_v.on_changed(on_v_change)
menu_btn.on_clicked(toggle_menu)


# --- Animation Callback ---
def animate(frame):
    current_k = slider_k.val

    # Physics sub-stepping for fast, responsive movement
    for _ in range(sub_steps):
        step(current_k)

    # 1. Update 3D Scatter
    ax_3d.clear()
    ax_3d.set_xlim(-bounds, bounds)
    ax_3d.set_ylim(-bounds, bounds)
    ax_3d.set_zlim(-bounds, bounds)
    ax_3d.set_title(f"3D Box (N = {len(atoms)})", fontsize=10)
    ax_3d.scatter(
        atoms[:, 0], atoms[:, 1], atoms[:, 2], color="crimson", s=25
    )

    # 2. Update Kinetic Energy
    ke = 0.5 * m * np.sum(atoms[:, 3:6] ** 2)
    elapsed = len(time_history) * (dt * sub_steps)
    time_history.append(elapsed)
    ke_history.append(ke)

    ke_line.set_data(time_history, ke_history)
    ax_ke.set_xlim(0, max(5.0, time_history[-1]))

    min_ke = min(ke_history)
    max_ke = max(ke_history)
    pad = max(1.0, (max_ke - min_ke) * 0.15)
    ax_ke.set_ylim(min_ke - pad, max_ke + pad)


anim = FuncAnimation(fig, animate, frames=600, interval=25)
plt.show()
