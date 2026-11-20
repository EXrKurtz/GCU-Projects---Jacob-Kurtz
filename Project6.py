# Jacob Kurtz

import math
import numpy as np
import matplotlib.pyplot as plt

def eval_poly(coeffs, x, x0=0.0):
    return sum(c * ((x - x0) ** k) for k, c in enumerate(coeffs))

# ==============================================================================
# PART 1(a): y'' - 2xy' + x^2y = 0, y(0) = 1, y'(0) = -1 (n <= 4, eval at 3.5)
# ==============================================================================
def taylor_coeffs_1a(N):
    derivs = {0: 1.0, 1: -1.0, 2: 0.0}
    if N >= 3:
        derivs.update({3: 2.0 * derivs.get(1)})
    for k in range(4, N + 1):
        val = 2 * (k - 2) * derivs.get(k - 2) - (k - 2) * (k - 3) * derivs.get(k - 4)
        derivs.update({k: val})
    return [derivs.get(k) / math.factorial(k) for k in range(N + 1)]

coeffs_1a = taylor_coeffs_1a(4)
val_1a_3_5 = eval_poly(coeffs_1a, 3.5)

print("--- PART 1(a) ---")
print(f"Taylor Coefficients (n=0..4): {[round(c, 5) for c in coeffs_1a]}")
print(f"P_4(3.5) = {val_1a_3_5:.6f}\n")

# Separate Window 1: Part 1(a)
plt.figure(1, figsize=(7, 5))
x_1a = np.linspace(-1.0, 3.8, 300)
plt.plot(x_1a, eval_poly(coeffs_1a, x_1a), color='tab:blue', lw=2, label=r'$f(x) = 1 - x - \frac{x^3}{3} - \frac{x^4}{12}$')
plt.scatter([3.5], [val_1a_3_5], color='red', zorder=5, label=f'y(3.5) = {val_1a_3_5:.2f}')
plt.title("Part 1(a): Taylor Polynomial ($n \\leq 4$)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True, alpha=0.3)


# ==============================================================================
# PART 1(b): y'' - (x - 2)y' + 2y = 0, y(3) = 6, y'(3) = 1 (2nd-order near x=3)
# ==============================================================================
x0_1b = 3.0
y0_1b, dy0_1b = 6.0, 1.0
d2y0_1b = (x0_1b - 2) * dy0_1b - 2 * y0_1b  # -11.0
coeffs_1b = [y0_1b, dy0_1b, d2y0_1b / math.factorial(2)]

print("--- PART 1(b) ---")
print("P_2(x) = 6 + (x - 3) - 5.5(x - 3)^2\n")

# Separate Window 2: Part 1(b)
plt.figure(2, figsize=(7, 5))
x_1b = np.linspace(1.5, 4.5, 300)
plt.plot(x_1b, eval_poly(coeffs_1b, x_1b, x0=3.0), color='tab:blue', lw=2, label=r'$f(x) = 6 + (x - 3) - \frac{11}{2}(x - 3)^2$')
plt.axvline(x=3.0, color='red', linestyle='--', lw=1.5, label='Expansion Point (x=3)')
plt.title("Part 1(b): 2nd-Order Taylor Polynomial")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True, alpha=0.3)


# ==============================================================================
# PART 2: (x^2 + 4)y'' + y = x around ordinary point x = 0 (n <= 9)
# ==============================================================================
def power_series_coeffs_2(N, a0=0.0, a1=0.0):
    a = {0: a0, 1: a1}
    if N >= 2:
        a.update({2: -a.get(0) / 8.0})
    if N >= 3:
        a.update({3: (1.0 - a.get(1)) / 24.0})
    for n in range(2, N - 1):
        val = -((n * (n - 1) + 1) / (4.0 * (n + 2) * (n + 1))) * a.get(n)
        a.update({n + 2: val})
    return [a.get(k) for k in range(N + 1)]

coeffs_2 = power_series_coeffs_2(9, a0=0.0, a1=0.0)

print("--- PART 2 ---")
print(f"Series Coefficients (n=0..5): {[round(c, 6) for c in coeffs_2[:6]]}")

# Separate Window 3: Part 2 matching the reference curve
plt.figure(3, figsize=(7, 5))
x_2 = np.linspace(0.5, 9.35, 300)
plt.plot(x_2, eval_poly(coeffs_2, x_2), color='tab:blue', lw=2, label='Power Series ($n \\leq 9$)')
plt.title("Part 2: Power Series Solution with $n \\leq 9$")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.ylim(-36000, 1500)
plt.xlim(0, 9.8)
plt.legend()
plt.grid(True, alpha=0.3)

# Display all 3 windows
plt.show()