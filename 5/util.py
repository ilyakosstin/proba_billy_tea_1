import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def walkers_scheme(prob_arr):
    n = len(prob_arr)

    u_arr = [n * pi for pi in prob_arr]
    k_arr = list(range(n))

    overfull_q = [i for i in range(n) if u_arr[i] > 1]
    underfull_q = [i for i in range(n) if u_arr[i] < 1]

    while underfull_q and overfull_q:
        overf = overfull_q.pop()
        underf = underfull_q.pop()

        k_arr[underf] = overf
        u_arr[overf] = u_arr[overf] + u_arr[underf] - 1

        if u_arr[overf] >= 1:
            overfull_q.append(overf)
        else:
            underfull_q.append(overf)

    while overfull_q:
        u_arr[overfull_q.pop()] = 1

    while underfull_q:
        u_arr[underfull_q.pop()] = 1
    
    return u_arr, k_arr


def generate_walkers(u=None, k=None, p=None):
    if p is not None:
        u, k = walkers_scheme(p)

    assert u is not None
    assert k is not None

    i = np.random.randint(0, len(u))
    w = np.random.uniform(0, 1)

    if w < u[i]:
        return i
    else:
        return k[i]


def emperical_probabilities_of_discrete(discrete_list):
    counter = {}

    for d in discrete_list:
        counter[d] = counter.get(d, 0) + 1

    return {
        k: v / len(discrete_list) for k, v in counter.items()
    }

def test_walkers_scheme(n=10, n_iter=100):
    weights = np.random.uniform(0, 1, size=n)
    weights = weights / weights.sum()

    orig = {
        i: wi for i, wi in enumerate(weights)
    }

    u, k = walkers_scheme(weights)
    
    walkers = emperical_probabilities_of_discrete([
        generate_walkers(u=u, k=k) for i in range(n_iter)
    ])

    orig_x = np.array(list(orig.keys()))
    walkers_x = np.array(list(walkers.keys()))

    walkers_h = np.array(list(walkers.values()))
    orig_h = np.array(list(orig.values()))


    width = 0.35

    plt.bar(orig_x - width/2, orig_h, width, color='red', label='orig')
    plt.bar(walkers_x + width/2, walkers_h, width, color='blue', label='walkers')
    plt.xticks(orig_x)

    plt.legend()
    plt.show()


def inverse(func, search_range, epsilon=1e-8):
    lo, hi = search_range
    def inv(t):
        l, r = lo, hi
        while r - l > epsilon:
            m = (r + l) / 2
            if func(m) < t:
                l = m
            else:
                r = m
        return r
    return inv


def plot_func(func, func_name, param_name, x_range, critical_points,
              n_side=200, zoom_factor=10, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    x_min, x_max = x_range
    pts = sorted(critical_points)

    for cp in pts:
        if x_min < cp:
            xs = np.linspace(x_min, cp, n_side * zoom_factor)
            ys = np.vectorize(func)(xs)
            ax.plot(xs, ys, label=f"{func_name}" if x_min == x_range[0] else "")
        y = func(cp)
        ax.scatter([cp], [y], color='red', zorder=5)
        x_min = cp

    if x_min < x_max:
        xs = np.linspace(x_min, x_max, n_side * zoom_factor)
        ys = np.vectorize(func)(xs)
        ax.plot(xs, ys)

    ax.set_xlabel(param_name)
    ax.set_ylabel(func_name)
    ax.grid(True)

