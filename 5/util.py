import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Схема Уолкера
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


# Функция для генерации индекса дискретной случайной величины
# можно дать u,k -- результат работы walkers_scheme
# а можно просто список вероятностей
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

# Подсчет эмпирических вероятностей по уже готовому
# списку сгенерированных значений
def emperical_probabilities_of_discrete(discrete_list):
    counter = {}

    for d in discrete_list:
        counter[d] = counter.get(d, 0) + 1

    return {
        k: v / len(discrete_list) for k, v in counter.items()
    }

# эмпирическая функция распределения
def build_distr_func(values):
    def F_generated(x):
        return sum(1 if v <= x else 0 for v in values) / len(values)
    return F_generated

# функция для тестирования схемы Уолкера, в блокноте запускается и выводит график
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

# Создание обратной функции с помощью бинарного поиска по интервалу
def inverse(func, search_range, epsilon=1e-10):
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


# утилита для рисования графиков функций
def plot_function(
    f,
    bounds,
    critical_points=None,
    ax=None,
    mu=1e-3,
    theta=0.1,
    base_step=0.01,
    fine_step=0.001, 
    parameter_name="x",
    function_name="f",
    color='royalblue'
):
    a, b = bounds

    f = np.vectorize(f)

    if critical_points is None:
        critical_points = []

    critical_points = np.asarray(critical_points, dtype=float)

    critical_points = critical_points[
        (critical_points > a) & (critical_points < b)
    ]

    critical_points = np.sort(critical_points)
    critical_points = np.unique(critical_points)

    segments = []
    left = a
    for c in critical_points:
        right = c - mu
        if left < right:
            segments.append((left, right))
        left = c + mu
    
    if left < b:
        segments.append((left, b))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    for left, right in segments:
        x_base = np.arange(left, right, base_step)
        fine_regions = []

        for c in critical_points:
            fine_left = max(left, c - theta)
            fine_right = min(right, c + theta)

            if fine_left < fine_right:
                fine_x = np.arange(fine_left, fine_right, fine_step)
                fine_regions.append(fine_x)

        if fine_regions:
            x = np.concatenate([x_base] + fine_regions)
        else:
            x = x_base

        x = np.unique(x)
        x = np.append(x, right)

        with np.errstate(all="ignore"):
            y = np.asarray(f(x), dtype=float)

        if y.ndim == 0:
            y = np.full(x.shape, y.item())

        valid = np.isfinite(y)
        indices = np.flatnonzero(valid)

        if len(indices) == 0:
            continue

        breaks = np.where(np.diff(indices) > 1)[0] + 1
        groups = np.split(indices, breaks)

        for group in groups:
            if len(group) > 0:
                ax.plot(x[group], y[group], color=color)

    ax.set_title(f"Graph of {function_name}({parameter_name})")
    ax.set_xlabel(parameter_name)
    ax.set_ylabel(f'{function_name}({parameter_name})')
    ax.grid(True)

    if ax is None:
        plt.tight_layout()
        plt.show()
