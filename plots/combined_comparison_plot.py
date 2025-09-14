import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from result_data import *

# make sure modular observed checkpoint is recorded at the end of each fragment exploration
modular_observed_cp = [
    [row[j] for j in range(len(row)) if j % 3 == 1] for row in modular_observed_cp
]

# check length of checkpoints, ensure they are the same
for i in range(len(modular_observed_cp)):
    temp = set([
        len(modular_observed_cp[i]),
        len(modular_roll_cp[i]),
        len(modular_time_cp[i]),
        len(naive_rollout_observed_cp[i]),
        len(naive_time_observed_cp[i]),
    ])
    assert len(temp) == 1, f"Length mismatch in run {i}"


# generate combined plot for rollout performed

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
plt.ticklabel_format(style="plain", axis="x")  # turn off sci-notation

for i, (obs, roll, naive) in enumerate(zip(modular_observed_cp, modular_roll_cp, naive_rollout_observed_cp)):

    x_modular = np.linspace(min(roll), max(roll), 100)
    poly_modular = np.poly1d(np.polyfit(roll, obs, deg=1))
    poly_naive = np.poly1d(np.polyfit(roll, naive, deg=1))

    ax.plot(x_modular, poly_modular(x_modular), 'r-', linewidth=1,
            label="modular fit" if i == 0 else None)
    ax.plot(x_modular, poly_naive(x_modular), 'b--', linewidth=1,
            label="naive fit" if i == 0 else None)

    ax.plot(roll, obs, 'ro', markersize=3,
            label="modular points" if i == 0 else None)
    ax.plot(roll, naive, 'bo', markersize=3,
            label="naive points" if i == 0 else None)

plt.xlabel("rollouts performed")
plt.xticks(fontsize=7)
plt.ylabel("% map explored")
plt.title("Expected Model Performance vs. Rollouts Performed")
plt.legend(fontsize=7)
plt.grid(True)

plt.savefig("ziheng_combined_rollout.png", dpi=300, bbox_inches="tight")
plt.close()


# generate combined plot for time elapsed

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
plt.ticklabel_format(style="plain", axis="x")  # turn off sci-notation

for i, (obs, time, naive_time) in enumerate(zip(modular_observed_cp, modular_time_cp, naive_time_observed_cp)):
    x_modular = np.linspace(min(time), max(time), 100)
    poly_modular = np.poly1d(np.polyfit(time, obs, deg=1))
    poly_naive = np.poly1d(np.polyfit(time, naive_time, deg=1))

    ax.plot(x_modular, poly_modular(x_modular), 'r-', linewidth=1,
            label="modular fit" if i == 0 else None)
    ax.plot(x_modular, poly_naive(x_modular), 'b--', linewidth=1,
            label="naive fit" if i == 0 else None)

    ax.plot(time, obs, 'ro', markersize=3,
            label="modular points" if i == 0 else None)
    ax.plot(time, naive_time, 'bo', markersize=3,
            label="naive points" if i == 0 else None)

plt.xlabel("time elapsed")
plt.xticks(fontsize=7)
plt.ylabel("% map explored")
plt.title("Expected Model Performance vs. Time Elapsed")
plt.legend(fontsize=7)
plt.grid(True)

plt.savefig("ziheng_combined_time.png", dpi=300, bbox_inches="tight")
plt.close()
