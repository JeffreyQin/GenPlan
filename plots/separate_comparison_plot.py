import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from result_data import *

# make sure modular observed checkpoint is recorded at the end of each fragment exploration
modular_observed_cp = result = [
    [row[j] for j in range(len(row)) if j % 3 == 1] for row in modular_observed_cp
]

# check length of checkpoints, ensure they are the same
for i in range(len(modular_observed_cp)):
    temp = set([len(modular_observed_cp[i]), len(modular_roll_cp[i]), len(modular_time_cp[i]), len(naive_rollout_observed_cp[i]), len(naive_time_observed_cp[i])])
    assert(len(temp) == 1)


# generate individual graph for rollout
for i in range(len(modular_observed_cp)):
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
    plt.ticklabel_format(style='plain', axis='x')  # turn off sci-notation

    x_smooth = np.linspace(min(modular_roll_cp[i]), max(modular_roll_cp[i]), 500)

    poly_red = np.poly1d(np.polyfit(modular_roll_cp[i], modular_observed_cp[i], deg=1))
    poly_blue = np.poly1d(np.polyfit(modular_roll_cp[i], naive_rollout_observed_cp[i], deg=1))

    # Plot fitted linear lines
    plt.plot(x_smooth, poly_red(x_smooth), 'r-', linewidth=1, label="modular fit")
    plt.plot(x_smooth, poly_blue(x_smooth), 'b--', linewidth=1, label="naive fit")

    # Scatter original data points
    plt.plot(modular_roll_cp[i], modular_observed_cp[i], 'ro', markersize=4, label="modular points")
    plt.plot(modular_roll_cp[i], naive_rollout_observed_cp[i], 'bo', markersize=4, label="naive points")

    # Labels and title
    plt.xlabel("rollouts performed")
    plt.xticks(fontsize=7)
    plt.ylabel("% map explored")
    plt.title("Expected Model Performance vs. Rollouts Performed")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"ziheng_{i+1}_rollout_vs_observed.png", dpi=300, bbox_inches="tight")
    plt.close()


# generate individual graph for time
for i in range(len(modular_observed_cp)):
    x_smooth = np.linspace(min(modular_time_cp[i]), max(modular_time_cp[i]), 500)

    poly_red = np.poly1d(np.polyfit(modular_time_cp[i], modular_observed_cp[i], deg=1))
    poly_blue = np.poly1d(np.polyfit(modular_time_cp[i], naive_time_observed_cp[i], deg=1))

    # Plot fitted linear lines
    plt.plot(x_smooth, poly_red(x_smooth), 'r-', linewidth=1, label="modular fit")
    plt.plot(x_smooth, poly_blue(x_smooth), 'b--', linewidth=1, label="naive fit")

    # Scatter original data points
    plt.plot(modular_time_cp[i], modular_observed_cp[i], 'ro', markersize=4, label="modular points")
    plt.plot(modular_time_cp[i], naive_time_observed_cp[i], 'bo', markersize=4, label="naive points")

    # Labels and title
    plt.xlabel("time elapsed (seconds)")
    plt.ylabel("% map explored")
    plt.title("Expected Model Performance vs. Time Elapsed")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"ziheng_{i+1}_time_vs_observed.png", dpi=300, bbox_inches="tight")
    plt.close()