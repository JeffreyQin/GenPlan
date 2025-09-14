import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from result_data import *


"""
generate benchmarking plot for rollout
"""

# extract the final explored percentage from each env
modular_roll = [sub[-1] for sub in modular_observed_cp]
naive_roll = [sub[-1] for sub in naive_rollout_observed_cp]

# x axis (env id)
x = np.arange(len(modular_observed_cp))  

fig, ax = plt.subplots(figsize=(6,4))

# comparison bars
width = 0.35
modular_bar = ax.bar(x - width/2, modular_roll, width, label="modular")
naive_bar = ax.bar(x + width/2, naive_roll, width, label="naive")

# Labels
ax.set_xlabel("environment")
ax.set_ylabel("# map explored")
ax.set_title("Naive Against Modular based on Rollouts Performed", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([str(i+1) for i in x])  # start labels at 1
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

ax.bar_label(modular_bar, labels=[f"{v.get_height()*100:.0f}%" for v in modular_bar], padding=3)
ax.bar_label(naive_bar, labels=[f"{v.get_height()*100:.0f}%" for v in naive_bar], padding=3)

plt.tight_layout()
plt.savefig(f"benchmark_rollout.png", dpi=300, bbox_inches="tight")



"""
generate benchmarking plot for time
"""

# extract the final explored percentage from each env
modular_time = [sub[-1] for sub in modular_observed_cp]
naive_time = [sub[-1] for sub in naive_time_observed_cp]

# x axis (env id)
x = np.arange(len(modular_observed_cp))  

fig, ax = plt.subplots(figsize=(6,4))

# comparison bars
width = 0.35
modular_bar = ax.bar(x - width/2, modular_time, width, label="modular")
naive_bar = ax.bar(x + width/2, naive_time, width, label="naive")

# Labels
ax.set_xlabel("environment")
ax.set_ylabel("# map explored")
ax.set_title("Naive Against Modular based on Time Elapsed", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels([str(i+1) for i in x])  # start labels at 1
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

ax.bar_label(modular_bar, labels=[f"{v.get_height()*100:.0f}%" for v in modular_bar], padding=3)
ax.bar_label(naive_bar, labels=[f"{v.get_height()*100:.0f}%" for v in naive_bar], padding=3)

plt.tight_layout()
plt.savefig(f"benchmark_time.png", dpi=300, bbox_inches="tight")
