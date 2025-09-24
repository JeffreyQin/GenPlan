import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from result_data import *

# make sure modular observed checkpoint is recorded at the end of each fragment exploration
modular_observed_cp = [
    [row[j] for j in range(len(row)) if j % 3 == 1] for row in modular_observed_cp
]

# check length of checkpoints, ensure they are the same
for i in range(len(modular_observed_cp)):
    temp = set([len(modular_observed_cp[i]), len(modular_roll_cp[i]), len(modular_time_cp[i]), len(naive_rollout_observed_cp[i]), len(naive_time_observed_cp[i])])
    assert(len(temp) == 1)


def extend_line(xi, yi, xmax_final):
    """Extend a line after the last point to reach 1.0 and then plateau."""
    if yi[-1] >= 1.0:
        return np.array([xi[-1], xmax_final]), np.array([1.0, 1.0])

    # slope from last two points
    if len(xi) > 1:
        slope = (yi[-1] - yi[-2]) / (xi[-1] - xi[-2])
    else:
        slope = 0.0

    x_ext = np.linspace(xi[-1], xmax_final, 500)
    y_ext = yi[-1] + slope * (x_ext - xi[-1])
    y_ext = np.minimum(y_ext, 1.0)

    plateau_index = np.argmax(y_ext >= 1.0)
    if y_ext[plateau_index] >= 1.0:
        y_ext[plateau_index:] = 1.0

    return x_ext, y_ext


# --- Custom scientific notation formatter ---
def sci_notation(x, _):
    """Format ticks like 1.2e8, 5e6 (no +, with sig figs)."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    base = x / 10**exp
    return f"{base:.2g}e{exp}"


# --- Generate individual graph for rollout ---
for i in range(len(modular_observed_cp)):
    plt.figure(figsize=(6, 6))

    # use sci notation formatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(sci_notation))
    plt.tick_params(axis='both', labelsize=15)

    xmax_final = max(modular_roll_cp[i]) * 1.05

    # --- Modular line ---
    xi = np.array(modular_roll_cp[i])
    yi = np.array(modular_observed_cp[i])
    plt.plot(xi, yi, 'r-', linewidth=2)
    plt.scatter(xi, yi, color='r', s=20, marker='o', alpha=0.7)
    x_ext, y_ext = extend_line(xi, yi, xmax_final)
    plt.plot(x_ext, y_ext, 'r-', linewidth=2)

    # --- Naive line ---
    xi_naive = np.array(modular_roll_cp[i])
    yi_naive = np.array(naive_rollout_observed_cp[i])
    plt.plot(xi_naive, yi_naive, 'b--', linewidth=2, dashes=(4, 4))
    plt.scatter(xi_naive, yi_naive, color='b', s=20, marker='x', alpha=0.7)
    x_ext_naive, y_ext_naive = extend_line(xi_naive, yi_naive, xmax_final)
    plt.plot(x_ext_naive, y_ext_naive, 'b--', linewidth=2, dashes=(4, 4))

    plt.xlabel("Rollouts performed", fontsize=15)
    plt.ylabel("Fraction explored", fontsize=15)
    plt.title(f"Gen-POMCP (Solid) v.s. Naive-POMCP (Dashed) - Env {i+1}", fontsize=13)
    plt.grid(True)

    plt.savefig(f"ziheng_{i+1}_rollout_vs_observed.png", dpi=300, bbox_inches="tight")
    plt.close()


# --- Generate individual graph for time ---
for i in range(len(modular_observed_cp)):
    plt.figure(figsize=(6, 6))

    # use sci notation formatter
    plt.gca().xaxis.set_major_formatter(FuncFormatter(sci_notation))
    plt.tick_params(axis='both', labelsize=15)

    xmax_final = max(modular_time_cp[i]) * 1.05

    # --- Modular line ---
    xi = np.array(modular_time_cp[i])
    yi = np.array(modular_observed_cp[i])
    plt.plot(xi, yi, 'r-', linewidth=2)
    plt.scatter(xi, yi, color='r', s=20, marker='o', alpha=0.7)
    x_ext, y_ext = extend_line(xi, yi, xmax_final)
    plt.plot(x_ext, y_ext, 'r-', linewidth=2)

    # --- Naive line ---
    xi_naive = np.array(modular_time_cp[i])
    yi_naive = np.array(naive_time_observed_cp[i])
    plt.plot(xi_naive, yi_naive, 'b--', linewidth=2, dashes=(4, 4))
    plt.scatter(xi_naive, yi_naive, color='b', s=20, marker='x', alpha=0.7)
    x_ext_naive, y_ext_naive = extend_line(xi_naive, yi_naive, xmax_final)
    plt.plot(x_ext_naive, y_ext_naive, 'b--', linewidth=2, dashes=(4, 4))

    plt.xlabel("Time elapsed (seconds)", fontsize=15)
    plt.ylabel("Fraction explored", fontsize=15)
    plt.title(f"Gen-POMCP (Solid) v.s. Naive-POMCP (Dashed) - Env {i+1}", fontsize=13)
    plt.grid(True)

    plt.savefig(f"ziheng_{i+1}_time_vs_observed.png", dpi=300, bbox_inches="tight")
    plt.close()
