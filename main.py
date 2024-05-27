import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_csv(filename) -> tuple:
        data = pd.read_csv(filename)
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        return np.array(x), np.array(y)


class LagrangeInterpolation:
    def __init__(self):
        pass

    @staticmethod
    def interpolate(x, x_interpolation, y_interpolation) -> float:
        result_value = 0
        for i in range(len(x_interpolation)):
            value = 1
            for j in range(len(x_interpolation)):
                if i != j and x_interpolation[i] != x_interpolation[j]:
                    small_random_number = random.uniform(-1e-6, 1e-6)
                    value *= (x - x_interpolation[j]) / (x_interpolation[i] - x_interpolation[j] + small_random_number)
            result_value += value * y_interpolation[i]
        return result_value


class CubicSplineInterpolation:
    def __init__(self):
        pass

    @staticmethod
    def cubic_spline_coefficients(x_original, y_original) -> tuple:
        n = len(x_original)
        h = [x_original[i + 1] - x_original[i] for i in range(n - 1)]

        a = np.zeros((n, n))
        b = np.zeros(n)

        # Set up the equations for the coefficients c
        for i in range(1, n - 1):
            a[i][i - 1] = h[i - 1]
            a[i][i] = 2 * (h[i - 1] + h[i])
            a[i][i + 1] = h[i]
            b[i] = 3 * ((y_original[i + 1] - y_original[i]) / h[i] - (y_original[i] - y_original[i - 1]) / h[i - 1])

        # Boundary conditions for natural spline (second derivative at endpoints are zero)
        a[0][0] = 1
        a[n - 1][n - 1] = 1

        c = np.linalg.solve(a, b)

        a_coefficient = y_original
        b_coefficient = [(y_original[i + 1] - y_original[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3 for i in range(n - 1)]
        d_coefficient = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n - 1)]

        # Since c has n elements but b and d have n-1 elements, we should only return the relevant parts
        return a_coefficient, b_coefficient, c, d_coefficient

    @staticmethod
    def interpolate(x_interpolation_points, y_interpolation_points, x_original) -> list:
        # Calculate spline coefficients using interpolation points
        a, b, c, d = CubicSplineInterpolation.cubic_spline_coefficients(x_interpolation_points, y_interpolation_points)
        n = len(x_interpolation_points)

        interpolated_values = []

        for x in x_original:
            # Find the right interval for x
            for i in range(n - 1):
                if x_interpolation_points[i] <= x <= x_interpolation_points[i + 1]:
                    break

            # Calculate the difference from the left point of the interval
            dx = x - x_interpolation_points[i]

            # Evaluate the spline polynomial
            interpolated_value = (a[i]
                                  + b[i] * dx
                                  + c[i] * dx ** 2
                                  + d[i] * dx ** 3)

            interpolated_values.append(interpolated_value)

        return interpolated_values


class ArrayMethods:
    def __init__(self):
        pass

    @staticmethod
    def linspace_int(start, stop, num) -> list:
        if num <= 0:
            return []
        if num == 1:
            return [start]

        step = (stop - start) / (num - 1)
        return [start + round(step * i) for i in range(num)]

    @staticmethod
    def chebyshev_indices(list_from, n_points) -> list:
        if n_points < 2:
            return [0]
        return [int((len(list_from) - 1) / 2 * (1 - np.cos(np.pi * i / (n_points - 1)))) for i in range(n_points)]


class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_simple(x, y, title, x_label, y_label):
        plt.style.use('dark_background')

        plt.plot(x, y, label='Original function', color='cyan')
        plt.title(title, color='white')
        plt.xlabel(x_label, color='white')
        plt.ylabel(y_label, color='white')
        plt.legend()

        plt.xticks(color='white')
        plt.yticks(color='white')

        plt.show()

    @staticmethod
    def plot_interpolated(x_data, y_data, x_new, y_new, title,
                          x_label, y_label):
        plt.style.use('dark_background')

        plt.plot(x_data, y_data, label='Data', color='orange')
        plt.plot(x_data, y_new, label='Interpolation result')
        plt.scatter([x_data[x] for x in x_new], [y_new[x] for x in x_new], color='green', label='Interpolation points')

        Plotter.add_common_parts(title, x_new, x_label, y_label)

    @staticmethod
    def plot_interpolated_spline(x_original,
                                 y_original,
                                 x_interpolation_nodes,
                                 y_interpolation_nodes,
                                 y_interpolation_function,
                                 x_label,
                                 y_label,
                                 title):
        plt.style.use('dark_background')

        plt.plot(x_original, y_original, label='Data', color='orange')
        plt.plot(x_original, y_interpolation_function, label='Interpolation result')
        plt.scatter(x_interpolation_nodes, y_interpolation_nodes, color='green', label='Interpolation points')

        Plotter.add_common_parts(title, x_interpolation_nodes, x_label, y_label)

    @staticmethod
    def add_common_parts(title, x_new, x_label, y_label):
        plt.title(f'{title} (n = {len(x_new)})', color='white')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()

        plt.xticks(color='white')
        plt.yticks(color='white')

        plt.show()


class Tester:
    def __init__(self, path):
        self.path = path

    def test_uniform_lagrange_interpolation(self, n_nodes: list):
        data_reader = DataReader()
        x, y = data_reader.read_csv(self.path)

        for n in n_nodes:
            uniform_idx = ArrayMethods.linspace_int(0, len(x) - 1, n)

            x_idx = x[uniform_idx]
            y_idx = y[uniform_idx]

            y_vals = [LagrangeInterpolation.interpolate(x[i], x_idx, y_idx) for i in range(len(x))]

            Plotter.plot_interpolated(x, y, uniform_idx, y_vals, 'Uniform Lagrange Interpolation', 'x', 'y')

    def test_chebyshev_lagrange_interpolation(self, n_nodes: list):
        data_reader = DataReader()
        x, y = data_reader.read_csv(self.path)

        for n in n_nodes:
            chebyshev_idx = ArrayMethods.chebyshev_indices(x, n)

            x_idx = x[chebyshev_idx]
            y_idx = y[chebyshev_idx]

            y_vals = [LagrangeInterpolation.interpolate(x[i], x_idx, y_idx) for i in range(len(x))]

            Plotter.plot_interpolated(x, y, chebyshev_idx, y_vals, 'Chebyshev Lagrange Interpolation',
                                      'x', 'y')

    def test_cubic_spline_interpolation(self, n_nodes: list):
        data_reader = DataReader()
        x, y = data_reader.read_csv(self.path)

        for n in n_nodes:
            indices_cubic = np.linspace(0, len(x) - 1, n, dtype=int)

            x_cubic = x[indices_cubic]
            y_cubic = y[indices_cubic]

            n_cubic = len(x_cubic)

            # Create system of equations

            n_equations = n_cubic - 1

            A = np.zeros((4 * n_equations, 4 * n_equations))
            b = np.zeros(4 * n_equations)
            p = np.zeros(4 * n_equations)

            # Fill up b vector

            b[0] = y_cubic[0]  # first point
            b[2 * n_equations - 1] = y_cubic[n_equations]  # last point

            for i in range(n_equations - 1):  # double points in the middle
                b[2 * i + 1] = y_cubic[i + 1]
                b[2 * i + 2] = y_cubic[i + 1]

            # Fill rest with zeros

            for i in range(2 * n_equations):
                p[2 * n_equations + i] = 0

            # Equals between ranges ( first 2 * n conditions )

            for i in range(n_equations):
                # f_i = f_i+1
                A[2 * i, 4 * i:4 * i + 4] = [x_cubic[i] ** 3, x_cubic[i] ** 2, x_cubic[i], 1]  # f_i
                A[2 * i + 1, 4 * i:4 * i + 4] = [x_cubic[i + 1] ** 3, x_cubic[i + 1] ** 2, x_cubic[i + 1], 1]  # f_i+1

            # First and second derivatives ( next 2 * (n - 1) conditions )

            for i in range(n_equations - 1):
                # d/dx (f_i) = d/dx (f_i+1)
                A[2 * n_equations + 2 * i, 4 * i:4 * i + 3] = [3 * x_cubic[i + 1] ** 2, 2 * x_cubic[i + 1],
                                                               1]  # d/dx (f_i)
                A[2 * n_equations + 2 * i, 4 * (i + 1):4 * (i + 1) + 3] = [-3 * x_cubic[i + 1] ** 2,
                                                                           -2 * x_cubic[i + 1],
                                                                           -1]  # -d/dx (f_i+1)

                # d^2/dx^2 (f_i) = d^2/dx^2 (f_i+1)
                A[2 * n_equations + 2 * i + 1, 4 * i:4 * i + 2] = [6 * x_cubic[i + 1], 2]  # d^2/dx^2 (f_i)
                A[2 * n_equations + 2 * i + 1, 4 * (i + 1):4 * (i + 1) + 2] = [-6 * x_cubic[i + 1],
                                                                               -2]  # -d^2/dx^2 (f_i+1)

            # 2 boundary conditions ( zeroing second derivatives at edges of the range )

            A[4 * n_equations - 2, 0:2] = [6 * x_cubic[0], 2]  # d^2/dx^2 (f_0) = 0
            A[4 * n_equations - 1, 4 * (n_equations - 1):4 * (n_equations - 1) + 2] = [6 * x_cubic[n_equations],
                                                                                       2]  # d^2/dx^2 (f_n) = 0

            # Solve system of equations

            p = np.linalg.solve(A, b)

            # Interpolated function values and plot

            cubic_functions = [[p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]] for i in range(n_equations)]

            # Interpolated values on original range

            y_interpolated_cubic = []

            for i in range(len(x)):
                for j in range(n_equations):
                    if x_cubic[j] <= x[i] <= x_cubic[j + 1]:
                        y_interpolated_cubic.append(cubic_functions[j][0] * x[i] ** 3 +
                                                    cubic_functions[j][1] * x[i] ** 2 +
                                                    cubic_functions[j][2] * x[i] +
                                                    cubic_functions[j][3])
                        break

            # Interpolated values on interpolation points

            y_inter_nodes_cubic = []

            for i in range(n_equations):
                y_inter_nodes_cubic.append(cubic_functions[i][0] * x_cubic[i] ** 3 +
                                           cubic_functions[i][1] * x_cubic[i] ** 2 +
                                           cubic_functions[i][2] * x_cubic[i] +
                                           cubic_functions[i][3])

            y_inter_nodes_cubic.append(cubic_functions[n_equations - 1][0] * x_cubic[n_equations] ** 3 +
                                       cubic_functions[n_equations - 1][1] * x_cubic[n_equations] ** 2 +
                                       cubic_functions[n_equations - 1][2] * x_cubic[n_equations] +
                                       cubic_functions[n_equations - 1][3])

            Plotter.plot_interpolated_spline(x, y, x_cubic, y_inter_nodes_cubic, y_interpolated_cubic, 'x', 'y',
                                             'Cubic Spline Interpolation')


def main():
    paths = [
        './2018_paths/WielkiKanionKolorado.csv',
        './2018_paths/MountEverest.csv',
        './2018_paths/SpacerniakGdansk.csv',
    ]

    tester = Tester(paths[0])
    #tester.test_uniform_lagrange_interpolation([8, 16])
    #tester.test_chebyshev_lagrange_interpolation([75])
    tester.test_cubic_spline_interpolation([15, 30, 45, 60, 75, 90])


if __name__ == '__main__':
    main()
