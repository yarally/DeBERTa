import math
import random
import sys
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm
import numpy as np


def random_optimization(dim, lower_bounds, upper_bounds, epochs_):
    y_best = sys.maxsize
    x_best = []
    random_samples = get_random_sample(dim, epochs_, lower_bounds, upper_bounds)
    for sample in random_samples:
        next_ = branin_hoo(sample)
        if next_ < y_best:
            y_best = next_
            x_best = sample
    return y_best, x_best[0], x_best[1]


def bayesian_optimization(dim, epochs_, lower_bounds, upper_bounds, warmup_=1, sample_size_=1000):
    X = get_random_sample(dim, warmup_, lower_bounds, upper_bounds)
    y = []
    for sample in X:
        y.append(branin_hoo(sample))

    model = GaussianProcessRegressor()
    for _ in tqdm(range(epochs_ - warmup_)):
        model.fit(X, y)
        candidate_samples = get_quasi_random_sample(2, sample_size_, lower_bounds, upper_bounds)
        X.append(max_p_improvement(candidate_samples, model, min(y)))
        y.append(branin_hoo(X[-1]))

    # plot_test_vs_predict(model, lower_bounds, upper_bounds)
    return min(y), X[y.index(min(y))][0], X[y.index(min(y))][1]


def plot_test_vs_predict(model, lower_bounds, upper_bounds):
    random_samples = get_random_sample(2, 1000, lower_bounds, upper_bounds)
    x_test = np.array(random_samples).transpose()[0]
    y_test = np.array(random_samples).transpose()[1]
    z_test = [branin_hoo(x) for x in random_samples]
    z_pred = [model.predict([x]) for x in random_samples]
    print('plotting graphs...')
    ax = plt.axes(projection='3d')
    ax.scatter(x_test, y_test, z_test, c='r', marker='o')
    ax.scatter(x_test, y_test, z_pred, c='b', marker='^')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.show()


def max_p_improvement(samples, model, y_min):
    best = -1
    x_next = None
    for x in samples:
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            mu_, std_ = model.predict([x], return_std=True)
            pi = norm.cdf((y_min - mu_) / (std_ + 5e-6))
            if pi > best:
                best = pi
                x_next = x
    return x_next


def branin_hoo(X):
    a = 1
    b = 5.1 / (4 * math.pi ** 2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)
    return a * (X[1] - b * X[0] ** 2 + c * X[0] - r) ** 2 + s * (1 - t) * math.cos(X[0]) + s


def get_quasi_random_sample(dim, length, lower, upper, sampler=qmc.Halton):
    sampler = sampler(dim)
    sample = sampler.random(length)
    return qmc.scale(sample, lower, upper)


def get_random_sample(dim, length, lower, upper):
    return [[random.uniform(lower[i], upper[i]) for i in range(dim)] for _ in range(length)]


if __name__ == '__main__':
    epochs = 64
    warmup = 1
    sample_size = 1000
    reps = 25
    print(f'Best possible: {branin_hoo([9.42478, 2.475])}')
    f = open("results.txt", "a")
    f.write(f'warmup={warmup},sample_size={sample_size},epochs={epochs}\n')
    f.close()
    for _ in range(reps):
        rnd, x1_rnd, x2_rnd = random_optimization(2, [-5, 0], [10, 15], epochs)
        bay, x1_bay, x2_bay = bayesian_optimization(2, epochs, [-5, 0], [10, 15], warmup_=warmup, sample_size_=sample_size)
        print(f'Best found for random: {rnd}, for x1={x1_rnd}, x2={x2_rnd}')
        print(f'Best found for bayesian: {bay}, for x1={x1_bay}, x2={x2_bay}')
        f = open("results.txt", "a")
        f.write(f'{rnd},{bay}\n')
        f.close()
