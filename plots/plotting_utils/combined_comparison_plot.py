import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
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


def extend_line(xi, yi, xmax_final):
    """Extend a line after last point to reach 1.0 and then plateau."""
    if yi[-1] >= 1.0:
        return np.array([xi[-1], xmax_final]), np.array([1.0, 1.0])

    # Slope from last two points
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


def plot_combined(x_data, modular_y, naive_y, xlabel, title, fname, xticks, tick_labelsize=15):
    """Plot connecting lines + extended plateau lines for both solid and dashed."""
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style="plain", axis="x")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    plt.xticks(rotation=xticks)
    plt.tick_params(axis='both', labelsize=tick_labelsize)  # bigger axis labels

    colors = plt.cm.tab10(np.linspace(0, 1, len(x_data)))
    xmax_global = max(max(r) for r in x_data)
    xmax_final = xmax_global * 1.05

    # Draw runs so later ones cover earlier plateaus
    order = sorted(range(len(x_data)), key=lambda i: x_data[i][-1])

    for i in order:
        # --- Modular solid line ---
        xi = np.array(x_data[i])
        yi = np.array(modular_y[i])
        plt.plot(xi, yi, '-', color=colors[i], linewidth=2)
        plt.scatter(xi, yi, marker='o', color=colors[i], s=20, alpha=0.7)

        x_ext, y_ext = extend_line(xi, yi, xmax_final)
        plt.plot(x_ext, y_ext, '-', color=colors[i], linewidth=2)

        # --- Naive dashed line ---
        xi_naive = np.array(x_data[i])
        yi_naive = np.array(naive_y[i])
        plt.plot(xi_naive, yi_naive, '--', color=colors[i], linewidth=2, dashes=(4, 4))
        plt.scatter(xi_naive, yi_naive, marker='x', color=colors[i], s=20, alpha=0.7)

        x_ext_naive, y_ext_naive = extend_line(xi_naive, yi_naive, xmax_final)
        plt.plot(x_ext_naive, y_ext_naive, '--', color=colors[i], linewidth=2, dashes=(4, 4))

    # Labels
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel("Fraction explored", fontsize=15)
    plt.title(title, fontsize=13)

    plt.ylim(0, 1.05 * 1.1)
    plt.xlim(0, xmax_final)

    # No legend
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# --- Generate plots ---
plot_combined(
    modular_roll_cp,
    modular_observed_cp,
    naive_rollout_observed_cp,
    xlabel="Rollouts performed",
    title="Exploration by Gen-POMCP (Solid) v.s. Naive-POMCP (Dashed)",
    fname="ziheng_combined_rollout.png",
    xticks=20,  # smaller tilt
    tick_labelsize=15
)

plot_combined(
    modular_time_cp,
    modular_observed_cp,
    naive_time_observed_cp,
    xlabel="Time elapsed (seconds)",
    title="Exploration by Gen-POMCP (Solid) v.s. Naive-POMCP (Dashed)",
    fname="ziheng_combined_time.png",
    xticks=20,
    tick_labelsize=15
)
