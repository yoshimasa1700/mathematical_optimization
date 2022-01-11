import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.001


def func(x):
    return (x - 2)**4 + x**2


def deriv_1st_func(x):
    return 4*(x - 2)**3 + 2 * x


def deriv_2nd_func(x):
    return 12*(x - 2)**2 + 2


def newton_method(func, deriv_1st_func, deriv_2nd_func, x0):
    x = x0

    x_hist = [x]

    safe_count = 0

    while abs(deriv_1st_func(x)) > EPSILON:

        safe_count +=1
        if safe_count >= 10:
            return x, x_hist

        d = - deriv_1st_func(x) / deriv_2nd_func(x)
        x = x + d
        x_hist.append(x)

    return x, x_hist


def visualize(func, start, end, ax, label, linestyle="solid"):
    x_list = np.arange(start, end, 0.01)
    y_list = [func(x) for x in x_list]
    ax.plot(x_list, y_list, label=label, linestyle=linestyle)


def main():

    # visualize original func
    fig = plt.figure(figsize=(10.0, 7.0))
    ax = fig.add_subplot()
    ax.grid()
    ax.set_ylim([0, 15.0])
    visualize(func, 0, 3.5, ax, "original function")

    # run newton method with initial guess
    x, x_hist = newton_method(func, deriv_1st_func, deriv_2nd_func, 3)

    # plot optimize hist
    ax.plot(x_hist, list(map(func, x_hist)),
            marker='x', markersize=15, label="optimize history")

    # plot approximated function
    for idx, x0 in enumerate(x_hist):
        q = lambda d: func(x0) + deriv_1st_func(x0) * (d - x0) + (1.0/2) * (d - x0) * deriv_2nd_func(x0) * (d - x0)
        visualize(q, 0, 3.5, ax, "{} th iteration".format(idx), "dotted")

    # plot optimal result
    ax.plot(x, func(x),
            marker='*', markersize=15, label="optimal result")

    fig.suptitle("Newton's optimization method")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
