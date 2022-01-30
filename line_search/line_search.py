import copy
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.001


def func(x):
    return (x - 2)**4 + x**2


def deriv_1st_func(x):
    return 4*(x - 2)**3 + 2 * x


tau = 0.2
alpha_step = 0.9

def armijo_condition(x0, alpha):
    dest_score = func(x0 - deriv_1st_func(x0) * alpha)
    armijo_score = func(x0) + deriv_1st_func(x0) * tau * alpha

    return armijo_score < dest_score


def line_search(func, deriv_1st_func, x0):
    x = x0

    x_hist = [x]


    safe_count = 0

    alpha = 0.4
    tau = 2.0

    alpha_hist = []

    while abs(deriv_1st_func(x)) > EPSILON:

        safe_count +=1
        if safe_count >= 10:
            return x, x_hist, alpha_hist

        # adjust alpha if not satisfy armijo condition.
        alpha_hist_temp = [alpha]
        while armijo_condition(x, alpha):
            alpha *= alpha_step
            alpha_hist_temp.append(alpha)

        d = -alpha * deriv_1st_func(x)
        x = x + d
        x_hist.append(x)
        alpha_hist.append(alpha_hist_temp)

    return x, x_hist, alpha_hist


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
    x, x_hist, alpha_step = line_search(func, deriv_1st_func, 3)

    # plot optimize hist
    ax.plot(x_hist, list(map(func, x_hist)),
            marker='x', markersize=15, label="optimize history")

    # plot armijo condition
    fig2 = plt.figure(figsize=(10.0, 7.0))
    ax2 = fig2.add_subplot()
    ax2.grid()
    # ax2.set_ylim([0, 30.0])

    for idx, x0 in enumerate([x_hist[0]]):
        g = lambda alpha: func(x0 - deriv_1st_func(x0) * alpha)
        q = lambda alpha: func(x0) + deriv_1st_func(x0) * tau * alpha

        visualize(g, 0, 0.41, ax2, "{} th iteration".format(idx))
        visualize(q, 0, 0.41, ax2, "{} th iteration".format(idx))

        ax2.plot(alpha_step[idx], list(map(g, alpha_step[idx])),
                marker='x', markersize=15, label="optimize history")


    # plot optimal result
    ax.plot(x, func(x),
            marker='*', markersize=15, label="optimal result")

    ax2.set_xlabel("alpha")
    ax2.set_ylabel("G(alpha)")
    fig2.suptitle("Armijo condition")

    fig.suptitle("Line search optimization method")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
