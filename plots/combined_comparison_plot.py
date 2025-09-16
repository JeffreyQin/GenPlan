import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from result_data import *
from scipy.stats import sem

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


# --- Combined plot for rollout performed ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
plt.ticklabel_format(style="plain", axis="x")  # turn off sci-notation

# Common x grid
x_common = np.linspace(
    min(min(r) for r in modular_roll_cp),
    max(max(r) for r in modular_roll_cp),
    100
)

# Interpolated values for each run
modular_interp = []
naive_interp = []
for obs, roll, naive in zip(modular_observed_cp, modular_roll_cp, naive_rollout_observed_cp):
    modular_interp.append(np.interp(x_common, roll, obs))
    naive_interp.append(np.interp(x_common, roll, naive))

modular_interp = np.array(modular_interp)
naive_interp = np.array(naive_interp)

# Mean and SEM
modular_mean = modular_interp.mean(axis=0)
modular_sem = sem(modular_interp, axis=0)
naive_mean = naive_interp.mean(axis=0)
naive_sem = sem(naive_interp, axis=0)

# Plot mean + 95% CI band
ax.plot(x_common, modular_mean, '-', color='red', linewidth=2, label="modular mean")
ax.fill_between(x_common, modular_mean - 1.96*modular_sem, modular_mean + 1.96*modular_sem,
                color='red', alpha=0.2, label="modular 95% CI")

ax.plot(x_common, naive_mean, '--', color='blue', linewidth=2, label="naive mean")
ax.fill_between(x_common, naive_mean - 1.96*naive_sem, naive_mean + 1.96*naive_sem,
                color='blue', alpha=0.2, label="naive 95% CI")

plt.xlabel("rollouts performed")
plt.xticks(fontsize=7)
plt.ylabel("% map explored")
plt.title("Expected Model Performance vs. Rollouts Performed")
plt.legend(fontsize=7)
plt.grid(True)

plt.savefig("ziheng_combined_rollout_CI.png", dpi=300, bbox_inches="tight")
plt.close()
