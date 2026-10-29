# Jacob Kurtz
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# define parameters
# Interpretation 1: Direct Flow Rates (V=1) -> Matrix coefficients match raw flows
# Interpretation 2: Volume-normalized (V=100) -> Scaling factor = 0.01
# Set interpretation = 1 or 2 below depending on your preference.

interpretation = 1

if interpretation == 1:
    A = np.array([[-2,  3],
                  [ 2, -3]])
    time_label = "Time (seconds)"
    t_end = 2.0
else:
    A = np.array([[-0.02,  0.03],
                  [ 0.02, -0.03]])
    time_label = "Time (seconds)"
    t_end = 200.0

# initial conditions: x1(0)=1, x2(0)=-1
x0 = np.array([1.0, -1.0])

# function for analytical solution

def analytical_solution(t, interp):
    if interp == 1:
        x1 = np.exp(-5 * t)
        x2 = -np.exp(-5 * t)
    else:
        x1 = np.exp(-0.05 * t)
        x2 = -np.exp(-0.05 * t)
    return x1, x2

# numerical solution for verification

def system_ode(t, x, A_mat):
    return A_mat @ x

t_eval = np.linspace(0, t_end, 500)
sol = solve_ivp(system_ode, [0, t_end], x0, args=(A,), t_eval=t_eval)

x1_analytical, x2_analytical = analytical_solution(t_eval, interpretation)

# plotting graph

plt.figure(figsize=(10, 6), dpi=100)

# lines
plt.plot(t_eval, x1_analytical, 'b-', linewidth=2.5, label='x1(t) [Analytical]')
plt.plot(t_eval, x2_analytical, 'r-', linewidth=2.5, label='x2(t) [Analytical]')

# verification points
plt.plot(sol.t, sol.y[0], 'k--', alpha=0.7, label='x1(t) [Numerical Verification]')
plt.plot(sol.t, sol.y[1], 'g:', alpha=0.7, label='x2(t) [Numerical Verification]')

# labels
plt.title(f'Data Degradation Simulation', fontsize=14, fontweight='bold')
plt.xlabel(time_label, fontsize=12)
plt.ylabel('Data / I/O Amount (MB)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')

# legend
plt.legend(loc='best', fontsize=10)
plt.tight_layout()

plt.savefig('data_degradation_part2.png', dpi=300)
plt.show()