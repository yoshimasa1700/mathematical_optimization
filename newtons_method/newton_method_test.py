import numpy as np
import matplotlib.pyplot as plt

from newton_method import EPSILON
from newton_method import newton_method
from newton_method import visualize


def func(x):
    return (1 + abs(x)) * np.log(1 + abs(x)) - (1 + abs(x))


def deriv_1st_func(x):

    if x >= 0:
        return np.log(1 + abs(x))
    else:
        return -np.log(1 + abs(x))


def deriv_2nd_func(x):
    return 1/(1 + abs(x))


def main():

    # visualize original func
    fig = plt.figure(figsize=(10.0, 7.0))
    ax = fig.add_subplot()
    ax.grid()
    ax.set_ylim([-30, 70.0])
    ax.set_xlim([-30, 30.0])
    visualize(func, -30, 30, ax, "original function")

    # run newton method with initial guess
    x, x_hist = newton_method(func, deriv_1st_func, deriv_2nd_func, np.e**2 - 1)

    # plot optimize hist
    ax.plot(x_hist, list(map(func, x_hist)),
            marker='x', markersize=15, label="optimize history")

    # plot approximated function
    for idx, x0 in enumerate(x_hist):
        q = lambda d: func(x0) + deriv_1st_func(x0) * (d - x0) + (1.0/2) * (d - x0) * deriv_2nd_func(x0) * (d - x0)
        visualize(q, -30, 30, ax, "{} th iteration".format(idx), "dotted")

    # plot optimal result
    ax.plot(x, func(x),
            marker='*', markersize=15, label="optimal result")

    fig.suptitle("Newton's optimization method")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
