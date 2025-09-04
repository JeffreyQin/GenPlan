import numpy as np
import matplotlib.pyplot as plt

# Solid red line data
roll_red = np.array([289662, 510571, 808812, 2155946, 2361049, 2843827, 5258429])

map_red = np.array([0.15217391304347827, 0.24456521739130435, 0.41304347826086957,
                    0.6956521739130435, 0.7880434782608695, 0.8586956521739131, 1.0])

# Dotted blue line data
roll_blue = np.array([289662, 510571, 808812, 2155946, 2361049, 2843827, 5258429])

map_blue = np.array([
    0.07173913043478261,
    0.07826086956521739,
    0.08521739130434783,
    0.09130434782608696,
    0.09652173913043478,
    0.09913043478260870,
    0.09978260869565217
])
# Create a smooth x-axis range for plotting
x_smooth = np.linspace(min(roll_red.min(), roll_blue.min()),
                       max(roll_red.max(), roll_blue.max()), 500)

# Fit polynomials (degree chosen small to avoid overfitting)
poly_red = np.poly1d(np.polyfit(roll_red, map_red, deg=3))
poly_blue = np.poly1d(np.polyfit(roll_blue, map_blue, deg=3))

# Plot scatter points
plt.scatter(roll_red, map_red, color='red')
plt.scatter(roll_blue, map_blue, color='blue')

# Plot fitted curves
plt.plot(x_smooth, poly_red(x_smooth), 'r-', linewidth=1, label="modular")
plt.plot(x_smooth, poly_blue(x_smooth), 'b--', linewidth=1, label="naive")

# Labels and title
plt.xlabel("rollouts performed")
plt.ylabel("% map explored")
plt.title("Expected Model Performance v.s. Rollouts Performed")
plt.legend()
plt.grid(True)

plt.savefig("map_9_rollout_vs_observed.png", dpi=300, bbox_inches="tight")

plt.show()

