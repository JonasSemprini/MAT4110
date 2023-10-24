import numpy as np
import pandas as pd
import random

matrix = np.array(
    [
        [5, 1 / np.sqrt(2), -1 / np.sqrt(2)],
        [1 / np.sqrt(2), 5 / 2, 7 / 2],
        [-1 / np.sqrt(2), 7 / 2, 5 / 2],
    ]
)

result = pd.DataFrame(columns=["Approx Eigenvalue", "Actual Eigenvalue", "Error"])

num_iterations = 10
actual_eigenvalues, _ = np.linalg.eig(matrix)

# print(actual_eigenvalues)
# actual_largest_eigenvalue = max(actual_eigenvalues)

# n = matrix.shape[0]
# b = np.random.rand(n)

# for i in range(num_iterations):
#     # Power iteration
#     b = np.dot(matrix, b)
#     # Normalize the vector
#     eigenvalue = np.linalg.norm(b)
#     error = abs(eigenvalue - actual_largest_eigenvalue)
#     b /= eigenvalue
#     result.loc[i] = [eigenvalue, actual_largest_eigenvalue, error]


# with pd.option_context(
#     "display.max_rows",
#     None,
#     "display.max_columns",
#     None,
#     "display.precision",
#     7,
# ):
#     print(result)


def inverse_power_method(matrix, mu, iter, tol=1e-15):
    Ashift = matrix - mu * np.identity(matrix.shape[0])
    b = np.zeros((len(Ashift), iter + 1))
    b[:, 0] = np.random.normal(0.5, 0.3, Ashift.shape[0])
    rn = np.ones((iter + 1,))
    for k in range(num_iterations):
        b[:, k] = b[:, k] / np.linalg.norm(b[:, k])
        b[:, k + 1] = np.linalg.solve(Ashift, b[:, k])
        rn[k + 1] = np.sum(b[:, k + 1]) / np.sum(b[:, k])
        if abs(rn[k + 1] - rn[k]) < tol:
            break
    if k < iter:
        rn[k + 2 :] = rn[k + 1]
    inv_pow = 1.0 / rn + mu
    for i in range(0, len(inv_pow)):
        eigenvalue = inv_pow[i]
        error = abs(eigenvalue - actual_eigenvalues[0])
        result.loc[i] = [eigenvalue, actual_eigenvalues[0], error]


mu = 5
inverse_power_method(matrix, mu, iter=num_iterations)

with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.precision",
    7,
):
    print(result)
