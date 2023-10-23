import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# results = pd.DataFrame(columns=["Iterations (n)", "Approx Integral"])

P_s = 2 ** (10)

n = [int(2 ** (1 + i)) for i in np.linspace(1, 8, 8)]


# Define the function to be integrated
def f(x, y):
    return np.exp(-((x - np.sin(y**2)) ** 3))


# Define the integration limits
x_min, x_max = 0, 1
y_min, y_max = 0, 1


integral_results = []


# Composite Trapezoidal Rule function
def composite_trapezoidal_rule(f, x_min, x_max, y_min, y_max, n):
    x_values = np.linspace(x_min, x_max, n)
    y_values = np.linspace(y_min, y_max, n)
    dx = (x_max - x_min) / n
    dy = (y_max - y_min) / n
    integral = 0.0
    for i in range(n):
        for j in range(n):
            xi = x_values[i]
            yi = y_values[j]
            xi1 = x_values[i + 1] if i < n - 1 else x_max
            yi1 = y_values[j + 1] if j < n - 1 else y_max
            integral += (
                0.25 * dx * dy * (f(xi, yi) + f(xi1, yi) + f(xi, yi1) + f(xi1, yi1))
            )
    return integral


# Calculate the integral for different values of n
for s in n:
    result = composite_trapezoidal_rule(f, x_min, x_max, y_min, y_max, s)
    integral_results.append(result)

# Print the results
# for i, s in enumerate(n):
#     result = integral_results[i]
#     results.loc[i] = [s, result]

# with pd.option_context(
#     "display.max_rows",
#     None,
#     "display.max_columns",
#     None,
#     "display.precision",
#     7,
# ):
#     print(results)


pseudo = composite_trapezoidal_rule(f, x_min, x_max, y_min, y_max, P_s)


error = [abs(pseudo - app) for app in integral_results]

print(len(error), len(n))
for i in range(1, len(n)):
    num = np.log(error[i]) - np.log(error[i - 1])
    den = np.log(n[i]) - np.log(n[i - 1])
    r = -(num / den)
    print(r)
