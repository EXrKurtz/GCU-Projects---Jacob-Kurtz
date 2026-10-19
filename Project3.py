#jacob kurtz

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- parameters ---
t_max = 15
fps = 30
frames = 300
t = np.linspace(0, t_max, frames)

# ODE 1 y'' + y = 4
y1_hom_cos = 4 * np.cos(t)
y1_hom_sin = 4 * np.sin(t)
y1_total = 4 * (1 - np.cos(t))  # Green's function

# ODE 2 y'' + 4y = x  (using t instead of x)
y2_hom_cos = 1 * np.cos(2 * t)
y2_hom_sin = 1 * np.sin(2 * t)
y2_total = 0.25 * t - 0.125 * np.sin(2 * t)  # Green's function

# --- plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Part 2: ODE Simulation & Verification", fontsize=14, fontweight='bold')

# ODE 1
ax1.set_xlim(0, t_max)
ax1.set_ylim(-5, 9)
ax1.set_ylabel("y(t)")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.set_title(r"ODE 1: $y'' + y = 4$ ($y(0)=y'(0)=0$)")

# ODE 2
ax2.set_xlim(0, t_max)
ax2.set_ylim(-0.5, 4)
ax2.set_xlabel("Time (t)")
ax2.set_ylabel("y(t)")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.set_title(r"ODE 2: $y'' + 4y = t$ ($y(0)=y'(0)=0$)")

# ODE 1 components
ax1.plot(t, y1_hom_cos, label="Homogeneous Basis 1: $4\\cos(t)$", color="orange", linestyle="--", alpha=0.5)
ax1.plot(t, y1_hom_sin, label="Homogeneous Basis 2: $4\\sin(t)$", color="green", linestyle=":", alpha=0.5)
line1, = ax1.plot([], [], label="Green's Function Solution", color="blue", linewidth=2.5)
point1, = ax1.plot([], [], 'ro')  # Current state tracker

# ODE 2 components
ax2.plot(t, y2_hom_cos, label="Homogeneous Basis 1: $\\cos(2t)$", color="orange", linestyle="--", alpha=0.5)
ax2.plot(t, y2_hom_sin, label="Homogeneous Basis 2: $\\sin(2t)$", color="green", linestyle=":", alpha=0.5)
line2, = ax2.plot([], [], label="Green's Function Solution", color="purple", linewidth=2.5)
point2, = ax2.plot([], [], 'ro')  # Current state tracker

ax1.legend(loc="upper left")
ax2.legend(loc="upper left")


# --- animation ---
def update(frame):
    line1.set_data(t[:frame], y1_total[:frame])
    point1.set_data([t[frame]], [y1_total[frame]])

    line2.set_data(t[:frame], y2_total[:frame])
    point2.set_data([t[frame]], [y2_total[frame]])

    return line1, point1, line2, point2


# --- run animation ---
ani = animation.FuncAnimation(
    fig, update, frames=frames, interval=1000 / fps, blit=True, repeat=True
)

plt.tight_layout()
plt.show()