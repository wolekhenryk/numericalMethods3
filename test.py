import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def cubic_spline_interpolation(x_values, y_values, x_new):
    # Perform cubic spline interpolation on the provided data points
    cs = CubicSpline(x_new, np.interp(x_new, x_values, y_values))

    # Compute y values for the original x_values using the cubic spline
    y_interpolated = cs(x_values)

    return y_interpolated


# Example usage
x_values = np.linspace(0, 10, 50)  # Original x values
y_values = np.sin(x_values)  # Original y values (example function output)
x_new = np.linspace(0, 10, 10)  # New x values for interpolation (subset of x_values)

y_interpolated = cubic_spline_interpolation(x_values, y_values, x_new)

# Display results
plt.plot(x_values, y_values, 'o', label='Original data')
plt.plot(x_new, np.interp(x_new, x_values, y_values), 'x', label='Interpolation nodes')
plt.plot(x_values, y_interpolated, '-', label='Cubic spline interpolation')
plt.legend()
plt.show()

# Print the interpolated values
print("Interpolated y values:", y_interpolated)
