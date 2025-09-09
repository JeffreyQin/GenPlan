import numpy as np
import matplotlib.pyplot as plt

# --- Original data ---
time_red = np.array([45.34574031829834, 260.2531666755676, 3187.3465135097504, 3394.732967853546, 6521.9108273983, 6723.6027953624725, 8798.463675022125, 11519.881461143494, 12008.487988233566, 12197.869036197662])

map_red = np.array([0.03934426229508197, 0.22459016393442624, 0.3377049180327869, 0.4163934426229508, 0.5295081967213114, 0.6081967213114754, 0.7213114754098361, 0.7934426229508197, 0.9131147540983606, 0.9918032786885246])

time_blue = np.array([45.34574031829834, 260.2531666755676, 3187.3465135097504, 3394.732967853546, 6521.9108273983, 6723.6027953624725, 8798.463675022125, 11519.881461143494, 12008.487988233566, 12197.869036197662])

map_blue = np.array([0.0014044943820224719, 0.0028089887640449437, 0.004213483146067416, 0.0056179775280898875, 0.0056179775280898875, 0.007022471910112359, 0.008426966292134831, 0.011235955056179775, 0.011235955056179775, 0.011235955056179775])

print(len(time_blue))
print(len(map_red))
print(len(map_blue))
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
plt.plot(time_red, map_red, 'ro', markersize=4, label="modular points")
plt.plot(time_blue, map_blue, 'bo', markersize=4, label="naive points")
# Labels and title
plt.xlabel("Time elapsed (seconds)")
plt.ylabel("% map explored")
plt.title("Expected Model Performance v.s. Time Elapsed")
plt.legend()
plt.grid(True)

plt.savefig("map_2_ziheng_time_vs_observed_poly.png", dpi=300, bbox_inches="tight")
plt.show()