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
print(actual_eigenvalues)
actual_largest_eigenvalue = max(actual_eigenvalues)

n = matrix.shape[0]
b = np.random.rand(n)

for i in range(num_iterations):
    # Power iteration
    b = np.dot(matrix, b)
    # Normalize the vector
    eigenvalue = np.linalg.norm(b)
    error = abs(eigenvalue - actual_largest_eigenvalue)
    b /= eigenvalue
    result.loc[i] = [eigenvalue, actual_largest_eigenvalue, error]


with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.precision",
    7,
):
    print(result)


# def inverse_power_method(A, mu, iter, tol=1e-15):
#     Ashift = A - mu * np.identity(A.shape[0])
#     b = np.zeros((len(A), iter + 1))
#     b[:, 0] = np.random.rand(A.shape[0])
#     print(b, b[0])
#     rn = np.ones((iter + 1,))
#     for k in range(num_iterations):
#         b[:, k] = b[:, k] / np.linalg.norm(b[:, k])
#         b[:, k + 1] = np.linalg.solve(Ashift, b[:, k])
#         rn[k + 1] = np.sum(b[:, k + 1]) / np.sum(b[:, k])
#         if abs(rn[k + 1] - rn[k]) < tol:
#             break
#     if k < iter:
#         rn[k + 2 :] = rn[k + 1]
#     return (
#         1.0 / rn[k + 1] + mu,
#         1.0 / rn + mu,
#         b[:, k + 1] / np.linalg.norm(b[:, k + 1]),
#     )


# lamda, v = np.linalg.eig(matrix)
# order = np.abs(lamda).argsort()
# lamda = lamda[order]
# mu = 2
# lamda_shift, lamda_seq, vpm = inverse_power_method(matrix, mu, iter=num_iterations)

# print(
#     "The eigenvalue closest to {} from the shifted power method is {} (exact is {}, error is {})".format(
#         mu, lamda_shift, lamda[1], abs(lamda_shift - lamda[1])
#     )
# )
