import numpy as np
import scipy.linalg as linalg
import sys
import numpy as np
import math

# def Newton(f, Df, x0, epsilon, max_iter):
#     xn = x0
#     for i in range(0, max_iter):
#         fxn = f(xn)
#         if abs(fxn) < epsilon:
#             print(f"Solution found after", {i}, "iterations.")
#             return xn
#         Dfxn = Df(xn)
#         if Dfxn == 0:
#             print("Derivative is zero. Undefined.")
#             return None
#         xn = xn - fxn / Dfxn
#     print("Exceeded maximum iterations. No solution found.")
#     return None


# def f(x):
#     return np.arctan(x)


# def dfdx(x):
#     return 1 / (1 + x**2)


# e = 1e-10
# x0 = 1.3
# n = Newton(f, dfdx, x0, e, 7)
# print(n)


# def f(x):
#     return np.arctan(x)


# def f_prime(x):
#     return 1 / (1 + x**2)


# def newton_method_arctan(x0, tolerance=1e-13, max_iterations=100):
#     x = x0
#     for i in range(max_iterations):
#         fx = f(x)
#         if abs(fx) < tolerance:
#             return x, i
#         x = x - fx / f_prime(x)
#     return None, max_iterations


# # Initial guess for the root
# x0 = 1.4

# # Call Newton's method to find the root
# root, iterations = newton_method_arctan(x0)

# if root is not None:
#     print(f"Approximate root: {root}")
#     print(f"Number of iterations: {iterations}")
# else:
#     print("Newton's method did not converge within the specified tolerance.")

# values = []
# non = []
# for i in x0:
#     n = Newton(g, dg, i, e, 30)
#     print(n)
#     if n == None:
#         non.append(i)
#     else:
#         values.append(i)
# print(max(values), min(non))

# # for _ in np.linspace(1, 10, 10):
# #     A = np.array([[1.0, 0, 1], [1, -1, 1]])
# #     B = np.array([2, 1, 3])
# #     Q, R = np.linalg.qr(A.T)  # QR decomposition with qr function
# #     y = np.dot(Q.T, B)  # Let y=Q'.B using matrix multiplication
# #     x_new = linalg.solve(R, y)  # Solve Rx=y
# #     x = x_new

A = np.array([[1.0, 0, 1], [1, -1, 1]])
B = np.array([2, 1, 3])
x = np.linalg.lstsq(A.T, B)
print(x[0])

# def g(x):
#     return ((x**2 + 1) * np.arctan(x)) / (x)


# def dg(x):
#     return 1 / x + np.arctan(x) - np.arctan(x) / (x**2)


# x_0 = 1.0
# epsilon = 1
# error = 1e-5
# x_n = 0

# while epsilon > error:
#     x_n = x_0 - (g(x_0) - 2) / dg(x_0)
#     epsilon = abs(x_n - x_0)
#     x_0 = x_n
# print(f"x_star: {x_0:.6f}")
