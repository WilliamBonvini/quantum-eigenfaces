from typing import Tuple

import numpy as np
from tqdm import tqdm

from quantum_eigenfaces.linalg import _compute_min_distance_to_label_mapping
from quantum_eigenfaces.misc import _compute_metrics_and_y_predict


def iterative_compute_metrics_and_y_predict(X_test: np.ndarray,
                                            y_test: np.ndarray,
                                            C_test: np.ndarray,
                                            y_train: np.ndarray,
                                            C_train: np.ndarray,
                                            delta1: float,
                                            epsilon: float,
                                            use_norm_threshold: bool,
                                            norm_threshold: float,
                                            norm_ratios_test: np.ndarray,
                                            num_iterations: int = 100,
                                            xi: float = 0.01,
                                            ) -> Tuple[Tuple, Tuple, np.ndarray]:

    # Initialization
    ACC = np.zeros((num_iterations,))
    ACC_TR = np.zeros((num_iterations,))
    FAR = np.zeros((num_iterations,))
    FRR = np.zeros((num_iterations,))
    PRE = np.zeros((num_iterations,))
    REC = np.zeros((num_iterations,))
    F1S = np.zeros((num_iterations,))
    Y_PREDICTS = np.zeros((num_iterations, len(X_test)), dtype="int")

    for k in tqdm(range(num_iterations), desc=f"performing {num_iterations} iterations", leave=False):

        # add noise corruption
        corrupted_min_distance_to_label_mapping = _compute_min_distance_to_label_mapping(C=C_test,
                                                                                         C_train=C_train,
                                                                                         y_train=y_train,
                                                                                         epsilon=epsilon)
        corrupted_norm_ratios_test = norm_ratios_test + np.random.uniform(-xi, xi, norm_ratios_test.shape)

        metrics, y_predict = _compute_metrics_and_y_predict(X_test=X_test,
                                                            y_test=y_test,
                                                            y_train=y_train,
                                                            min_distance_to_label_mapping=corrupted_min_distance_to_label_mapping,
                                                            delta1=delta1,
                                                            use_norm_threshold=use_norm_threshold,
                                                            norm_threshold=norm_threshold,
                                                            norm_ratios_test=corrupted_norm_ratios_test)
        accuracy, accuracy_train, false_acceptance_rate, false_recognition_rate, precision, recall, f1_score = metrics
        # historicize metrics for this iteration
        ACC[k] = accuracy
        ACC_TR[k] = accuracy_train
        FAR[k] = false_acceptance_rate
        FRR[k] = false_recognition_rate
        PRE[k] = precision
        REC[k] = recall
        F1S[k] = f1_score
        Y_PREDICTS[k] = y_predict

    # compute average metrics over all iterations
    avg_acc = np.mean(ACC)
    print(f"Avg Accuracy: {avg_acc}")
    avg_acc_tr = np.mean(ACC_TR)
    print(f"Avg Accuracy Train: {avg_acc_tr}")
    avg_far = np.mean(FAR)
    avg_frr = np.mean(FRR)
    avg_pre = np.mean(PRE)
    avg_rec = np.mean(REC)
    print(f"Avg Recall: {avg_rec}")
    avg_f1s = np.mean(F1S)
    metrics = avg_acc, avg_acc_tr, avg_far, avg_frr, avg_pre, avg_rec, avg_f1s

    # compute stddev of metrics over all iterations
    stddev_acc = np.std(ACC)
    stddev_acc_tr = np.std(ACC_TR)
    stddev_far = np.std(FAR)
    stddev_frr = np.std(FRR)
    stddev_pre = np.std(PRE)
    stddev_rec = np.std(REC)
    stddev_f1s = np.std(F1S)

    stddevs = stddev_acc, stddev_acc_tr, stddev_far, stddev_frr, stddev_pre, stddev_rec, stddev_f1s

    # compute predicted labels with majority voting
    y_predict = np.zeros((len(X_test),))
    for i in range(len(X_test)):
        u, indices = np.unique(Y_PREDICTS[:, i], return_inverse=True)
        y_predict[i] = u[np.argmax(np.bincount(indices))]

    return metrics, stddevs, y_predict
