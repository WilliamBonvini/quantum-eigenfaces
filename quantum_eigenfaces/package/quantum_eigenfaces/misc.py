from typing import Tuple, Optional

import numpy as np


def _compute_metrics_and_y_predict(X_test: np.ndarray,
                                   y_test: np.ndarray,
                                   y_train: np.ndarray,
                                   min_distance_to_label_mapping: np.ndarray,
                                   delta1: float,
                                   use_norm_threshold: bool,
                                   norm_threshold: Optional[float],
                                   norm_ratios_test: Optional[np.ndarray]) -> Tuple[Tuple[float, ...], np.ndarray]:

    assert not use_norm_threshold or norm_ratios_test is not None
    assert not use_norm_threshold or norm_threshold is not None

    # initialize to zero metrics
    tot_samples = len(X_test)
    correct_pred = 0
    true_anomaly = 0
    true_recognition = 0
    false_recognition = 0
    false_acceptance = 0
    num_test_ok = 0
    num_test_anomaly = 0
    false_anomaly = 0

    y_predict = np.zeros((len(X_test),), dtype="int")

    for i in range(len(X_test)):

        # classify
        if use_norm_threshold and norm_ratios_test[i] < norm_threshold:
            prediction = -3  # cut but norm threshold

        else:
            prediction = _classify(min_distance_to_label_mapping=min_distance_to_label_mapping,
                                   # back to corrupted if needed!!
                                   i=i,
                                   delta1=delta1,
                                   delta2=10 ** 6)
        y_predict[i] = prediction
        y = y_test[i]

        # update metrics count
        if y not in y_train:
            num_test_anomaly += 1
            if prediction < 0:
                correct_pred += 1
                true_anomaly += 1
            else:
                false_acceptance += 1
        else:
            num_test_ok += 1
            if prediction == y:
                correct_pred += 1
                true_recognition += 1
            else:
                if prediction == -1 or prediction == -3:
                    false_anomaly += 1
                else:
                    false_recognition += 1

    # Final metrics
    accuracy = correct_pred / tot_samples
    accuracy_train = true_recognition / num_test_ok if num_test_ok != 0 else 0

    denom = true_anomaly + false_anomaly
    precision = true_anomaly / denom if denom != 0 else 0

    denom = true_anomaly + false_acceptance
    recall = true_anomaly / denom if denom != 0 else 0

    denom = precision + recall
    f1_score = 2 * precision * recall / denom if denom != 0 else 0

    false_acceptance_rate = false_acceptance / tot_samples

    false_rejection_rate = false_anomaly / tot_samples

    metrics = accuracy, accuracy_train, false_acceptance_rate, false_rejection_rate, precision, recall, f1_score

    return metrics, y_predict


def _classify(min_distance_to_label_mapping: np.ndarray,
              i: int,
              delta1: float,
              delta2: float) -> int:
    d = min_distance_to_label_mapping[i, 0]
    if d <= delta1:
        training_label = min_distance_to_label_mapping[i, 1]
        return training_label

    if delta1 < d <= delta2:
        return -1

    if d >= delta2:
        return -2

