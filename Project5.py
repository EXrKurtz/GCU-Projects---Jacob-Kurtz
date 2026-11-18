# Jacob Kurtz

import numpy as np
import matplotlib.pyplot as plt


def filesystem_lorenz(x, y, z, r, s=10.0, b=2.667):
    """
    Dynamical system modeling deterministic chaos in file system fragmentation:
      x: Fragmentation index (non-contiguous block scatter)
      y: I/O latency stress (seek & allocation contention)
      z: Cumulative disk overhead & fragment assembly load
    """
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


# --- 1. File System & Simulation Parameters ---
dt = 0.01
num_steps = 10000
time_steps = np.arange(num_steps + 1) * dt

storage_size = 1000.0        # Total storage capacity (MB)
avg_file_size = 35.71        # Average file size (MB) -> yields r ≈ 28.0
io_churn_rate = 10.0         # 's': Frequency of save/delete commands
cleanup_rate = 2.667         # 'b': Background gap consolidation rate
#r = round(storage_size / avg_file_size, 2)  # Storage pressure factor (28.0)
r = 10

# Base hardware timing constants (ms)
base_access_time = 2.0
base_load_time = 5.0
base_save_time = 6.0

# Criticality threshold (ms) for total file save latency
critical_threshold = 42.0

# --- 2. State Arrays ---
xs = np.empty(num_steps + 1)  # Fragmentation index
ys = np.empty(num_steps + 1)  # I/O latency stress
zs = np.empty(num_steps + 1)  # Disk overhead & assembly load

# File system metric tracking arrays
file_sizes = np.empty(num_steps + 1)
fragmentation_time = np.empty(num_steps + 1)
fragments_assembly_time = np.empty(num_steps + 1)
file_access_time = np.empty(num_steps + 1)
file_load_time = np.empty(num_steps + 1)
file_save_time = np.empty(num_steps + 1)

# Initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# --- 3. Simulate Sequence of Save & Delete Commands ---
alert_triggered = False

for i in range(num_steps + 1):
    # Deterministic sequence of save and delete commands
    command = "SAVE" if np.sin(0.5 * time_steps[i]) >= -0.2 else "DELETE"
    current_file_size = avg_file_size * (1.0 + 0.3 * np.cos(0.3 * time_steps[i]))
    file_sizes[i] = current_file_size

    # Derive file system timing metrics (ms) from dynamical state (x, y, z)
    fragmentation_time[i] = 0.45 * abs(xs[i])
    fragments_assembly_time[i] = 0.35 * zs[i]
    file_access_time[i] = base_access_time + 0.25 * abs(ys[i])
    file_load_time[i] = base_load_time + file_access_time[i] + fragments_assembly_time[i]
    file_save_time[i] = (
        base_save_time
        + file_access_time[i]
        + fragmentation_time[i]
        + fragments_assembly_time[i]
    )

    # Check Self-Organized Criticality threshold
    if not alert_triggered and file_save_time[i] >= critical_threshold:
        alert_triggered = True
        print(
            f"[ALERT] SYSTEM TOO SLOW at t = {time_steps[i]:.2f}s | "
            f"Command: {command} | File Size: {current_file_size:.1f} MB | "
            f"Save Time: {file_save_time[i]:.2f} ms (Threshold: {critical_threshold} ms)"
        )

    # Step through time using partial derivatives
    if i < num_steps:
        x_dot, y_dot, z_dot = filesystem_lorenz(
            xs[i], ys[i], zs[i], r, s=io_churn_rate, b=cleanup_rate
        )
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

# --- FIGURE 1: 3D Lorenz Attractor Trajectory ---
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.plot(xs, ys, zs, lw=0.5, color='royalblue')
ax1.set_xlabel("X: Fragmentation Index")
ax1.set_ylabel("Y: I/O Latency Stress")
ax1.set_zlabel("Z: Disk Assembly Overhead")
ax1.set_title("File System Lorenz Attractor (3D) | r=" + str(r))

# --- FIGURE 2: X Component vs Time ---
fig2 = plt.figure()
ax_x = fig2.add_subplot(1, 1, 1)
ax_x.plot(time_steps, xs, lw=0.8, color='royalblue')
ax_x.set_xlabel("Time (t)")
ax_x.set_ylabel("X: Fragmentation Index")
ax_x.set_title("File System X Component (Fragmentation)")
ax_x.grid(True, linestyle='--', alpha=0.5)

# --- FIGURE 3: Y Component vs Time ---
fig3 = plt.figure()
ax_y = fig3.add_subplot(1, 1, 1)
ax_y.plot(time_steps, ys, lw=0.8, color='royalblue')
ax_y.set_xlabel("Time (t)")
ax_y.set_ylabel("Y: I/O Latency Stress")
ax_y.set_title("File System Y Component (I/O Latency)")
ax_y.grid(True, linestyle='--', alpha=0.5)

# --- FIGURE 4: Z Component vs Time ---
fig4 = plt.figure()
ax_z = fig4.add_subplot(1, 1, 1)
ax_z.plot(time_steps, zs, lw=0.8, color='royalblue')
ax_z.axhline(critical_threshold, color='crimson', linestyle='--', lw=0.8)
ax_z.set_xlabel("Time (t)")
ax_z.set_ylabel("Z: Disk Assembly Overhead")
ax_z.set_title("File System Z Component (Overhead & Critical Threshold)")
ax_z.grid(True, linestyle='--', alpha=0.5)

# Display all standalone window figures simultaneously
plt.show()