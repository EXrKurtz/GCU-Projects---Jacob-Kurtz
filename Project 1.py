
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# 1. Define the Workload Function W(t)
def cpu_workload(t):
    """
    Simulates a dynamic user workload over time (0 to 100% utilization).
    - 0 to 20s: Idle background tasks (10%)
    - 20s to 70s: Heavy compilation/gaming spike (90%)
    - 70s to 120s: Mixed usage recovery (40%)
    """
    if t < 20:
        return 10.0
    elif t < 70:
        return 90.0
    else:
        return 40.0


# 2. Define the ODE system
def cpu_thermal_ode(t, T, k, T_amb, alpha):
    """
    Calculates the rate of change of CPU temperature: dT/dt
    """
    W = cpu_workload(t)
    dT_dt = -k * (T - T_amb) + alpha * W
    return dT_dt


def main():
    print("--- CPU Thermal Dynamics Simulation ---")

    # System Parameters (Can be modified or requested via user input)
    T_amb = 35.0  # Ambient case temperature in °C
    T0 = 40.0  # Initial CPU temperature in °C at t=0
    k = 0.05  # Cooling coefficient (fan speed/heatsink efficiency)
    alpha = 0.04  # Heat generation coefficient per % workload

    t_start = 0.0
    t_end = 120.0  # Total simulation time in seconds
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, 500)  # Points where we evaluate the solution

    # 3. Solve the ODE numerically using RK45 (Runge-Kutta 4th/5th order)
    solution = solve_ivp(
        cpu_thermal_ode,
        t_span,
        [T0],
        args=(k, T_amb, alpha),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,  # Relative error tolerance
        atol=1e-8  # Absolute error tolerance
    )

    # Extract time and temperature results
    time_steps = solution.t
    cpu_temp = solution.y[0]

    # Calculate corresponding workloads for plotting tracking
    workloads = [cpu_workload(t) for t in time_steps]

    # 4. Visualization Paradigm (Dual-axis layout for clarity)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Axis: CPU Temperature
    color = '#d62728'
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('CPU Temperature (°C)', color=color, fontsize=12)
    line1 = ax1.plot(time_steps, cpu_temp, color=color, linewidth=2.5, label='CPU Temperature (°C)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(30, 95)

    # Secondary Axis: CPU Workload (Contextualization)
    ax2 = ax1.twinx()
    color = '#1f77b4'
    ax2.set_ylabel('CPU Utilization (%)', color=color, fontsize=12)
    line2 = ax2.plot(time_steps, workloads, color=color, linewidth=1.5, linestyle='--',
                     label='Workload Utilization (%)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)

    # Visual Enhancements
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Computer System Performance: CPU Thermal Response to Dynamic Workload', fontsize=14, fontweight='bold',
              pad=15)
    fig.tight_layout()

    # Display the visualization
    plt.show()


if __name__ == "__main__":
    main()