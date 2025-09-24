import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from matplotlib.ticker import FuncFormatter
from result_data import *

# --- Preprocess modular checkpoints (sampled every 3 steps) ---
modular_observed_cp = [
    [row[j] for j in range(len(row)) if j % 3 == 1] for row in modular_observed_cp
]

# --- Check consistency of run lengths ---
for i in range(len(modular_observed_cp)):
    lengths = {
        len(modular_observed_cp[i]),
        len(modular_roll_cp[i]),
        len(modular_time_cp[i]),
        len(naive_rollout_observed_cp[i]),
        len(naive_time_observed_cp[i]),
    }
    assert len(lengths) == 1, f"Length mismatch in run {i}"

# --- Helper: compute mean and 95% CI ---
def compute_mean_ci(data_list):
    arr = np.array(data_list)  # shape: (num_runs, num_checkpoints)
    mean = np.mean(arr, axis=0)
    sems = sem(arr, axis=0)
    ci95 = sems * t.ppf((1 + 0.95) / 2., arr.shape[0] - 1)
    return mean, ci95

# --- Improved scientific notation formatter ---
def sci_notation(x, _):
    """Format ticks like 1.2e8, 2.5e4 (no +, with sig figs)."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    base = x / 10**exp
    return f"{base:.2g}e{exp}"  # 2 significant digits

# --- Function to generate summary plot ---
def plot_summary(x_data_list, modular_y_list, naive_y_list, xlabel, fname):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Use scientific notation with significant digits
    ax.xaxis.set_major_formatter(FuncFormatter(sci_notation))
    plt.tick_params(axis='both', labelsize=15)

    # Interpolate all runs onto a common x grid
    x_common = np.linspace(0, max(max(r) for r in x_data_list) * 1.05, 500)

    # Modular (red)
    modular_interp = [np.interp(x_common, xi, yi) for xi, yi in zip(x_data_list, modular_y_list)]
    modular_mean, modular_ci = compute_mean_ci(modular_interp)
    plt.plot(x_common, modular_mean, 'r-', linewidth=2, label='Gen-POMCP')
    plt.fill_between(x_common, modular_mean - modular_ci, modular_mean + modular_ci, color='r', alpha=0.15)

    # Naive (blue, dashed)
    naive_interp = [np.interp(x_common, xi, yi) for xi, yi in zip(x_data_list, naive_y_list)]
    naive_mean, naive_ci = compute_mean_ci(naive_interp)
    plt.plot(x_common, naive_mean, 'b--', linewidth=2, label='Naive-POMCP', dashes=(6, 2))
    plt.fill_between(x_common, naive_mean - naive_ci, naive_mean + naive_ci, color='b', alpha=0.15)

    # Labels & style
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Fraction explored", fontsize=15)
    plt.title("Average Exploration with 95% Confidence Interval", fontsize=15)
    plt.ylim(0, 1.1)
    plt.xlim(0, x_common[-1])
    plt.grid(False)
    plt.legend(fontsize=10)
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

# --- Generate rollout summary plot ---
plot_summary(modular_roll_cp, modular_observed_cp, naive_rollout_observed_cp,
             xlabel="Rollouts performed", fname="ziheng_summary_rollout.png")

# --- Generate time summary plot ---
plot_summary(modular_time_cp, modular_observed_cp, naive_time_observed_cp,
             xlabel="Time elapsed (seconds)", fname="ziheng_summary_time.png")
