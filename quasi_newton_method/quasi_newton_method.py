import numpy as np
import matplotlib.pyplot as plt

from func import func
from func import deriv_1st_func


EPSILON = 0.001
tau = 0.3
alpha_step = 0.9


def armijo_condition(x0, alpha):
    arg = x0 - deriv_1st_func(x0[0, 0], x0[1, 0]) * alpha

    dest_score = func(arg[0, 0], arg[1, 0])

    def g(alpha):
        arg = x0 + deriv_1st_func(x0[0, 0], x0[1, 0]) * alpha
        return func(arg[0, 0], arg[1, 0])

    def g_dash(alpha):
        arg = x0 + deriv_1st_func(x0[0, 0], x0[1, 0]) * alpha
        return deriv_1st_func(arg[0, 0], arg[1, 0]).T.dot(-deriv_1st_func(x0[0, 0], x0[1, 0]))

    def q(alpha):
        return g(0) + tau * g_dash(0) * alpha

    armijo_score = q(alpha)

    return armijo_score > dest_score


def quasi_newton_method_BFGS(func, deriv_1st_func, x0):
    # define initial quasi matrix B
    B = np.eye(x0.shape[0])

    # init alpha
    alpha = 1.

    # init x
    x = x0
    x_prev = None
    deriv_1st_f = deriv_1st_func(x[0, 0], x[1, 0])

    x_hist = [x.T[0].tolist()]

    safe_count = 0

    while np.linalg.norm(deriv_1st_f) > EPSILON:
        safe_count += 1
        if safe_count >= 10:
            return x, x_hist

        B_inv = np.linalg.inv(B)

        d = - B_inv.dot(deriv_1st_f)

        # line search
        while not armijo_condition(x, alpha):
            alpha *= alpha_step

        x_prev = x
        x = x + d * alpha

        deriv_1st_f = deriv_1st_func(x[0, 0], x[1, 0])
        x_hist.append(x.T[0].tolist())

        # calc and update initial quasi matrix B
        s = d
        y = deriv_1st_func(x[0, 0], x[1, 0]) - \
            deriv_1st_func(x_prev[0, 0], x_prev[1, 0])

        B_s = B.dot(s)

        # print(B_s)

        term1_scalar = s.T.dot(B_s)

        # print("term1_scalar")
        # print(term1_scalar)

        term1_matrix = B_s.dot(B_s.T)
        term1 = term1_matrix / term1_scalar

        # print(term1)

        term2_scalar = s.T.dot(y)
        term2_matrix = y.dot(y.T)
        term2 = term2_matrix / term2_scalar

        B = B - term1 + term2

    return x, x_hist


def main():

    # define func and range.
    x = np.arange(-1, 5, 0.1)
    y = np.arange(-1, 5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # define initial value.
    x0 = np.array([[0., 3.]]).T

    # run quasi newton method.
    x, x_hist = quasi_newton_method_BFGS(func, deriv_1st_func, x0)

    print("x and x hist")
    print(x, x_hist)

    # visualize result.
    fig = plt.figure(figsize=(15, 10))
    plt.rcParams["font.size"] = 24
    ax = fig.add_subplot()
    ax.grid()

    def gen_lev(start, stop, step):
        return list(np.arange(start, stop, step))

    levels = gen_lev(0, 1, 0.5) + gen_lev(1, 5, 1) + gen_lev(6, 100, 10)
    ax.contourf(X, Y, Z, levels=levels, cmap="jet", alpha=0.1)
    cntr = ax.contour(X, Y, Z, levels=levels, alpha=0.5)
    ax.clabel(cntr)

    x_hist = np.array(x_hist)
    ax.plot(x_hist[:, 0], x_hist[:, 1], marker="o")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    fig.suptitle("Quasi newton method(BFGS)")
    fig.savefig("quasi_newton_method.png")

    plt.show()


if __name__ == "__main__":
    main()
