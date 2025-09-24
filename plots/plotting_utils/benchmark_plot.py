import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
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

# --- Colors for environments ---
colors = plt.cm.tab10(np.linspace(0, 1, len(modular_observed_cp)))

# --- Helper: extend line to plateau at 1.0 ---
def extend_line(xi, yi, xmax_final):
    if yi[-1] >= 1.0:
        return np.array([xi[-1], xmax_final]), np.array([1.0, 1.0])
    slope = (yi[-1] - yi[-2]) / (xi[-1] - xi[-2]) if len(xi) > 1 else 0.0
    x_ext = np.linspace(xi[-1], xmax_final, 500)
    y_ext = yi[-1] + slope * (x_ext - xi[-1])
    y_ext = np.minimum(y_ext, 1.0)
    plateau_index = np.argmax(y_ext >= 1.0)
    if y_ext[plateau_index] >= 1.0:
        y_ext[plateau_index:] = 1.0
    return x_ext, y_ext

# --- Improved scientific notation formatter ---
def sci_notation(x, _):
    """Format ticks like 1.2e8, 2.5e4 (no +, with sig figs)."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    base = x / 10**exp
    return f"{base:.2g}e{exp}"  # keep 2 significant digits

# --- Line plots ---
def plot_line_only(x_data, modular_y, naive_y, xlabel, title_line, fname_line):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Formatter for scientific notation
    ax.xaxis.set_major_formatter(FuncFormatter(sci_notation))
    # Locator: consistent tick spacing, ~6 ticks max
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    plt.tick_params(axis='both', labelsize=15)
    xmax_final = max(max(r) for r in x_data) * 1.05

    order = sorted(range(len(x_data)), key=lambda i: x_data[i][-1])

    for i in order:
        # Gen-POMCP (solid)
        xi = np.array(x_data[i])
        yi = np.array(modular_y[i])
        plt.plot(xi, yi, '-', color=colors[i], linewidth=2)
        plt.scatter(xi, yi, marker='o', color=colors[i], s=20, alpha=0.7)
        x_ext, y_ext = extend_line(xi, yi, xmax_final)
        plt.plot(x_ext, y_ext, '-', color=colors[i], linewidth=2)

        # Naive-POMCP (dashed)
        xi_naive = np.array(x_data[i])
        yi_naive = np.array(naive_y[i])
        plt.plot(xi_naive, yi_naive, '--', color=colors[i], linewidth=2, dashes=(4, 4))
        plt.scatter(xi_naive, yi_naive, marker='x', color=colors[i], s=20, alpha=0.7)
        x_ext_naive, y_ext_naive = extend_line(xi_naive, yi_naive, xmax_final)
        plt.plot(x_ext_naive, y_ext_naive, '--', color=colors[i], linewidth=2, dashes=(4, 4))

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Fraction explored", fontsize=15)
    plt.title(title_line, fontsize=13)
    plt.ylim(0, 1.1)
    plt.xlim(0, xmax_final)
    plt.savefig(fname_line, dpi=300, bbox_inches='tight')
    plt.close()

# --- Bar plots ---
def plot_bar_only(final_naive, xlabel, title_bar, fname_bar):
    order = np.argsort(final_naive)[::-1]
    naive_sorted = [final_naive[i] for i in order]
    x = np.arange(len(naive_sorted))
    width = 0.7

    fig, ax = plt.subplots(figsize=(5, 5))
    naive_bar = ax.bar(x, naive_sorted, width, color=[colors[i] for i in order])

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Fraction explored", fontsize=15)
    ax.set_title(title_bar, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"env{order[i]+1}" for i in range(len(order))], 
                       rotation=45, ha="right", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0, max(naive_sorted)*1.2)
    ax.bar_label(
        naive_bar,
        labels=[f"{v.get_height()*100:.0f}%" for v in naive_bar],
        padding=5,
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(fname_bar, dpi=300, bbox_inches='tight')
    plt.close()

# --- Rollout ---
plot_line_only(
    modular_roll_cp,
    modular_observed_cp,
    naive_rollout_observed_cp,
    xlabel="Rollouts performed",
    title_line="Exploration by Gen-POMCP (Solid) v.s. Naive-POMCP (Dashed)",
    fname_line="ziheng_combined_rollout.png"
)
final_naive_roll = [sub[-1] for sub in naive_rollout_observed_cp]
plot_bar_only(final_naive_roll, xlabel="Environment", 
              title_bar="Explored by Naive POMCP", 
              fname_bar="benchmark_rollout.png")

# --- Time ---
plot_line_only(
    modular_time_cp,
    modular_observed_cp,
    naive_time_observed_cp,
    xlabel="Time elapsed (seconds)",
    title_line="Exploration by Gen-POMCP (Solid) v.s. Naive-POMCP (Dashed)",
    fname_line="ziheng_combined_time.png"
)
final_naive_time = [sub[-1] for sub in naive_time_observed_cp]
plot_bar_only(final_naive_time, xlabel="Environment", 
              title_bar="Explored by Naive POMCP (Time Fixed)", 
              fname_bar="benchmark_time.png")
