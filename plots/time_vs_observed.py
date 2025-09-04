import numpy as np
import matplotlib.pyplot as plt

# Solid red line data
time_red = np.array([2.357133388519287, 8.692197322845459, 17.63937020301819,
                     63.173462867736816, 68.82655572891235, 76.70810651779175,
                     140.44598150253296])

map_red = np.array([0.15217391304347827, 0.24456521739130435, 0.41304347826086957,
                    0.6956521739130435, 0.7880434782608695, 0.8586956521739131, 1.0])

# Dotted blue line data
time_blue = np.array([2.361021041870117, 8.69641399383545, 17.645902156829834,
                      63.17918300628662, 68.83024787902832, 76.71158289909363,
                      140.44953322410583])

map_blue = np.array([0.07065217391304347, 0.07608695652173914, 0.07608695652173914,
                     0.11413043478260869, 0.11413043478260869, 0.11413043478260869,
                     0.11413043478260869])

# Create a smooth x-axis range for plotting
x_smooth = np.linspace(min(time_red.min(), time_blue.min()),
                       max(time_red.max(), time_blue.max()), 500)

# Fit polynomials (degree chosen small to avoid overfitting)
poly_red = np.poly1d(np.polyfit(time_red, map_red, deg=3))
poly_blue = np.poly1d(np.polyfit(time_blue, map_blue, deg=3))

# Plot scatter points
plt.scatter(time_red, map_red, color='red')
plt.scatter(time_blue, map_blue, color='blue')

# Plot fitted curves
plt.plot(x_smooth, poly_red(x_smooth), 'r-', linewidth=1, label="modular")
plt.plot(x_smooth, poly_blue(x_smooth), 'b--', linewidth=1, label="naive")

# Labels and title
plt.xlabel("Time elapsed (seconds)")
plt.ylabel("% map explored")
plt.title("Expected Model Performance v.s. Time Elapsed")
plt.legend()
plt.grid(True)

plt.savefig("map_9_time_vs_observed.png", dpi=300, bbox_inches="tight")

plt.show()

