import numpy as np
import matplotlib.pyplot as plt

# --- Original data ---
time_red = np.array([2.357133388519287, 8.692197322845459, 17.63937020301819,
 63.173462867736816, 68.82655572891235, 76.70810651779175,
 140.44598150253296, 156.57704489135742, 172.70810828018188,
 188.83917166900635, 204.9702350578308, 221.10129844665527])

map_red = np.array([0.1521739130434783, 0.2445652173913043, 0.4130434782608696,
 0.6956521739130435, 0.7880434782608695, 0.8586956521739131,
 0.97, 0.9723809523809523, 0.9845454545454546,
 0.9845217391304348, 0.983333333333333, 0.978930481283422])

time_blue = np.array([2.361021041870117, 8.69641399383545, 17.645902156829834,
 63.17918300628662, 68.83024787902832, 76.71158289909363,
 140.44953322410583, 156.58055990982056, 172.71158659553528,
 188.84261328125, 204.97363996696472, 221.10466665267944])

map_blue = np.array([0.0706521739130435, 0.0760869565217391, 0.0760869565217391,
 0.1141304347826087, 0.1141304347826087, 0.1141304347826087,
 0.1141304347826087, 0.1141304347826087, 0.1141304347826087,
 0.1141304347826087, 0.1141304347826087, 0.1141304347826087])

# --- Smooth x-axis ---
x_smooth = np.linspace(min(time_red.min(), time_blue.min()),
                       max(time_red.max(), time_blue.max()), 500)

# --- Fit cubic polynomials ---
poly_red = np.poly1d(np.polyfit(time_red, map_red, deg=3))
poly_blue = np.poly1d(np.polyfit(time_blue, map_blue, deg=3))

# --- Evaluate and clip red so it never exceeds 1 ---
y_red = np.clip(poly_red(x_smooth), 0, 1.0)
y_blue = poly_blue(x_smooth)

# --- Plot ---
plt.plot(x_smooth, y_red, 'r-', linewidth=1, label="modular")
plt.plot(x_smooth, y_blue, 'b--', linewidth=1, label="naive")

# Labels and title
plt.xlabel("Time elapsed (seconds)")
plt.ylabel("% map explored")
plt.title("Expected Model Performance v.s. Time Elapsed")
plt.legend()
plt.grid(True)

plt.savefig("map_9_time_vs_observed_poly.png", dpi=300, bbox_inches="tight")
plt.show()