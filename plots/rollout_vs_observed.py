import numpy as np
import matplotlib.pyplot as plt

# --- Original data ---
roll_red = np.array([2155540, 6993993, 46530120, 50322533, 91181118, 94955413, 121758569, 163665812, 168871709, 172743499])

map_red = np.array([0.03934426229508197, 0.22459016393442624, 0.3377049180327869, 0.4163934426229508, 0.5295081967213114, 0.6081967213114754, 0.7213114754098361, 0.7934426229508197, 0.9131147540983606, 0.9918032786885246])

roll_blue = np.array([2155540, 6993993, 46530120, 50322533, 91181118, 94955413, 121758569, 163665812, 168871709, 172743499])

map_blue = np.array([0.0014044943820224719, 0.0028089887640449437, 0.0028089887640449437, 0.0028089887640449437, 0.0028089887640449437, 0.0028089887640449437, 0.0028089887640449437, 0.004213483146067416, 0.0056179775280898875, 0.008426966292134831])

# Create a smooth x-axis range for plotting
x_smooth = np.linspace(min(roll_red.min(), roll_blue.min()),
                       max(roll_red.max(), roll_blue.max()), 500)

# Fit polynomials (degree chosen small to avoid overfitting)
poly_red = np.poly1d(np.polyfit(roll_red, map_red, deg=3))
poly_blue = np.poly1d(np.polyfit(roll_blue, map_blue, deg=3))

# Plot fitted curves
plt.plot(x_smooth, poly_red(x_smooth), 'r-', linewidth=1, label="modular")
plt.plot(x_smooth, poly_blue(x_smooth), 'b--', linewidth=1, label="naive")
plt.plot(roll_red, map_red, 'ro', markersize=4, label="modular points")
plt.plot(roll_blue, map_blue, 'bo', markersize=4, label="naive points")
# Labels and title
plt.xlabel("rollouts performed")
plt.ylabel("% map explored")
plt.title("Expected Model Performance v.s. Rollouts Performed")
plt.legend()
plt.grid(True)

plt.savefig("map_2_ziheng_rollout_vs_observed.png", dpi=300, bbox_inches="tight")

plt.show()

