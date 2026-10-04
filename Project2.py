import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import time

x, y = sp.symbols('x y')
#     --- ODE ---
expr = -(y**2)/x

# --- Parameters ---
x0 = 1.0
y0 = 3.0
x_end = 50.0 #steps
h = 0.01 #step size

my_ode = sp.lambdify((x, y), expr, 'numpy')

f_func = sp.Function('y')(x)
sympy_ode = sp.Eq(f_func.diff(x), expr.subs(y, f_func))
analytical_sol = sp.dsolve(sympy_ode, f_func, ics={f_func.subs(x, x0): y0})
exact_curve_generator = sp.lambdify(x, analytical_sol.rhs, 'numpy')

def rk4_step(f, x_val, y_val, h_step):
    k1 = f(x_val, y_val)
    k2 = f(x_val + h_step / 2, y_val + h_step / 2 * k1)
    k3 = f(x_val + h_step / 2, y_val + h_step / 2 * k2)
    k4 = f(x_val + h_step, y_val + h_step * k3)

    next_y = y_val + (h_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_y, (k1, k2, k3, k4)


def rk4_solve(f, x0, y0, x_end, h):
    num_steps = int((x_end - x0) / h)
    x_vals = np.linspace(x0, x0 + num_steps * h, num_steps + 1)
    y_vals = np.zeros(num_steps + 1)
    y_vals[0] = y0

    print(f"{'Step':<5} | {'x_val':<7} | {'y_val':<10} | {'k1':<10} | {'k2':<10} | {'k3':<10} | {'k4':<10}")
    print("-" * 80)

    for i in range(num_steps):
        # Calculate the next step and grab the current step's k-values
        y_vals[i + 1], ks = rk4_step(f, x_vals[i], y_vals[i], h)

        # Print current values formatted cleanly to 4 decimal places
        print(
            f"{i:<5} | {x_vals[i]:<7.4f} | {y_vals[i]:<10.4f} | {ks[0]:<10.4f} | {ks[1]:<10.4f} | {ks[2]:<10.4f} | {ks[3]:<10.4f}")

    # Print final endpoint evaluation row
    print(
        f"{num_steps:<5} | {x_vals[-1]:<7.4f} | {y_vals[-1]:<10.4f} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}\n")

    return x_vals, y_vals

start_time = time.perf_counter()

x, y = rk4_solve(my_ode, x0, y0, x_end, h)
x_exact = np.linspace(x0, x_end, 100)
y_exact = exact_curve_generator(x_exact)

# computational time
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Computational Time: {execution_time:.6f} seconds")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: RK4 Only
axes[0].plot(x, y, 'o-', label='RK4 Approx', color='blue')
axes[0].set_title('1. Runge-Kutta Approximation')
axes[0].grid(True)

# Plot 2: Exact ODE Solution
axes[1].plot(x_exact, y_exact, '-', label='Exact Solution', color='red')
axes[1].set_title('2. Auto Exact ODE Solution')
axes[1].grid(True)

# Plot 3: Both Overlapped
axes[2].plot(x, y, 'o', label='RK4 Points', color='blue')
axes[2].plot(x_exact, y_exact, '-', label='Exact Curve', color='red', alpha=0.8)
axes[2].set_title('3. Overlapped Comparison')
axes[2].grid(True)

plt.tight_layout()
plt.show()