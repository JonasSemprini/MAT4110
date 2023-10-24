import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

results = pd.DataFrame(columns=["Iterations (n)", "Approx Integral"])

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
def composite_trapezoidal_rule(f, n):
    x_values = np.linspace(0, 1, n + 1)
    y_values = np.linspace(0, 1, n + 1)
    h = 1 / n
    X, Y = np.meshgrid(x_values, y_values)
    z = f(X, Y)
    sum = np.sum(z[1:, 1:] + z[:-1, 1:] + z[1:, :-1] + z[:-1, :-1])
    comp_sum = ((h**2) / 4) * sum
    return comp_sum


# Calculate the integral for different values of n
for s in n:
    result = composite_trapezoidal_rule(f, s)
    integral_results.append(result)


for i, s in enumerate(n):
    result = integral_results[i]
    results.loc[i] = [s, result]

with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.precision",
    7,
):
    print(results)


pseudo = composite_trapezoidal_rule(f, P_s)


error = [abs(pseudo - app) for app in integral_results]

print(len(error), len(n))
for i in range(1, len(n)):
    num = np.log(error[i]) - np.log(error[i - 1])
    den = np.log(n[i]) - np.log(n[i - 1])
    r = -(num / den)
    print(r)
