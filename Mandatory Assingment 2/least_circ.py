import scipy.io
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("Solarize_Light2")

measurementData = scipy.io.loadmat("circle-measurements.mat")
x = measurementData["x"].reshape(-1)
y = measurementData["y"].reshape(-1)

phi = np.linspace(0, 1, 100)

b_1 = (max(x) + min(x)) / 2
b_2 = (max(y) + min(y)) / 2
b_3 = (abs(max(x)) + abs(min(x)) + abs((max(y)) + abs(min(x)))) / 4
beta = [b_1, b_2, b_3]

n = 3
matrix = np.zeros((len(x), n))


def circle_model(beta, x, y):
    return (x - beta[0]) ** 2 + (y - beta[1]) ** 2 - beta[2] ** 2


def Jacobian(matrix, beta):
    jac = matrix
    for i in range(0, len(x)):
        jac[i][0] = -2 * (x[i] - beta[0])
        jac[i][1] = -2 * (y[i] - beta[1])
        jac[i][2] = -2 * (beta[2])
    return jac


def gauss_newton(beta, max_iter=100, tolerance=1e-6):
    params = beta
    for _ in range(max_iter):
        J = Jacobian(matrix, params)
        r = circle_model(params, x, y)
        J_T = np.transpose(J)
        S = np.linalg.pinv(np.matmul(J_T, J))
        delta_params = np.matmul(S, np.matmul(J_T, r))

        # print(delta_params.shape, r.shape)
        params = params - delta_params

        if np.linalg.norm(delta_params) < tolerance:
            break

    return params


beta_n = gauss_newton(beta)
print(beta_n)

s1 = beta_n[0] + beta_n[2] * np.cos(2 * np.pi * phi)
s2 = beta_n[1] + beta_n[2] * np.sin(2 * np.pi * phi)
plt.scatter(x, y, marker="o", color="blue", label="Measure points")
plt.plot(s1, s2, color="red", label="Circle fit")
plt.legend()
plt.savefig("circle_fit.pdf")
plt.show()


# A = np.c_[2 * x, 2 * y, np.ones_like(x)]

# r = np.linalg.solve(A.T @ A, A.T @ (-(x**2) - (y**2)))

# r_fit = np.sqrt(r[0] ** 2 + r[1] ** 2 - r[2])

# print(r_fit)
