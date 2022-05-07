import numbers
import numpy as np
import matplotlib.pyplot as plt


def is_scalar(x):
    return isinstance(x, numbers.Number)


def func(X, Y):
    return (X - 2)**4 + (X - 2*Y)**2


def deriv_1st_func(X, Y):
    delta_x = 4 * (X - 2)**3 + 2 * (X - 2*Y)
    delta_y = -4 * (X - 2*Y)

    if is_scalar(delta_x):
        return np.array([[delta_x, delta_y]]).T

    return np.concatenate([delta_x, delta_y], 0)


def main():
    x = np.arange(-1, 5, 0.1)
    y = np.arange(-1, 5, 0.1)

    X, Y = np.meshgrid(x, y)

    print(X)

    print(Y)

    Z = func(X, Y)

    print(Z)

    def gen_lev(start, stop, step):
        return list(np.arange(start, stop, step))

    levels = gen_lev(0, 1, 0.5) + gen_lev(1, 5, 1) + gen_lev(6, 100, 10)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()

    ax.contourf(X, Y, Z, levels=levels, cmap="jet", alpha=0.1)
    cntr = ax.contour(X, Y, Z, levels=levels, alpha=0.5)
    ax.clabel(cntr)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    fig.suptitle("Original function")

    plt.show()


if __name__ == "__main__":
    main()
