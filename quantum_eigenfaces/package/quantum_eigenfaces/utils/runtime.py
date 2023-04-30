from typing import Union

import numpy as np
import gc

import pandas as pd


def __mu(p, matrix):
    def s(p, A):
        if p == 0:
            result = np.max([np.count_nonzero(A[i]) for i in range(len(A))])
        else:
            norms = np.sum(np.power(np.abs(A), p), axis=1)
            result = max(norms)
            del norms
        gc.collect()
        return result

    s1 = s(2 * p, matrix)
    s2 = s(2 * (1 - p), matrix.T)
    mu = np.sqrt(s1 * s2)

    gc.collect()
    return mu


def linear_search(matrix, start=0.0, end=1.0, step=0.05):
    domain = [i for i in np.arange(start, end, step)] + [end]
    values = [__mu(i, matrix) for i in domain]
    best_p = domain[values.index(min(values))]
    return best_p, min(values)


def compute_best_mu(matrix, start=0.0, end=1.0, step=0.05):
    p, val = linear_search(matrix, start=start, end=end, step=step)
    val_list = [val, np.linalg.norm(matrix)]
    index = val_list.index(min(val_list))
    if index == 0:
        best_norm = f"p={p}"
    elif index == 1:
        best_norm = "Frobenius"

    return best_norm, val_list[index]


def compute_quantum_runtime(sqrt_p: float,
                            mu: float,
                            max_c_train_norm: float,
                            U_bar: np.ndarray,
                            epsilon: float) -> np.ndarray:

    if epsilon == 0:
        return np.ones(U_bar.shape[1]) * (-1)

    U_bar_norms = np.linalg.norm(U_bar, axis=1)

    print(f"sqrt p: {sqrt_p}")
    print(f"mu: {mu}")
    print(f"max_c_train_norm: {max_c_train_norm}")
    print(f"avg U_bar_norms: {np.mean(U_bar_norms)}")
    print(f"epsilon: {epsilon}")
    return sqrt_p * mu * max_c_train_norm * U_bar_norms / epsilon


def find_highest_acceptable_epsilon(summary: pd.DataFrame) -> float:
    starting_accuracy = summary.loc[0, "Accuracy"]
    print(starting_accuracy)
    accuracy_threshold = starting_accuracy * 0.975
    print(accuracy_threshold)
    highest_acceptable_epsilon = summary["epsilon"][summary["Accuracy"] >= accuracy_threshold][-1:].values[0]
    return highest_acceptable_epsilon
