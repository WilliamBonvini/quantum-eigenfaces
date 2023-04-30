# ---------------- TUNING -------------------

from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from quantum_eigenfaces.linalg import _compute_min_distance_to_label_mapping
from quantum_eigenfaces.misc import _compute_metrics_and_y_predict


def tuning(
           y_train,
           C_train,
           X_valid,
           y_valid,
           C_valid,
           tot_num_of_deltas: int,
           use_norm_threshold: bool,
           norm_threshold: float,
           norm_ratios_valid: np.ndarray
           ) -> Tuple[float, Dict]:
    """ find value for chosen_delta1 """

    min_distance_to_label_mapping = _compute_min_distance_to_label_mapping(
        C=C_valid,
        C_train=C_train,
        y_train=y_train
    )

    delta1_domain_range = _compute_delta1_domain_range(
        min_distance_to_label_mapping=min_distance_to_label_mapping,
        tot_num_of_deltas=tot_num_of_deltas
    )

    accuracy_list, FAR_list, FRR_list, precision_list, recall_list, f1_score_list = (np.zeros(len(delta1_domain_range)) for _ in range(6))
    chosen_delta1 = -1
    best_accuracy = 0

    for i, delta1 in enumerate(tqdm(delta1_domain_range, desc="tuning delta1...")):
        metrics, _ = _compute_metrics_and_y_predict(
            X_test=X_valid,
            y_test=y_valid,
            y_train=y_train,
            min_distance_to_label_mapping=min_distance_to_label_mapping,
            delta1=delta1,
            use_norm_threshold=use_norm_threshold,
            norm_threshold=norm_threshold,
            norm_ratios_test=norm_ratios_valid
        )
        accuracy, accuracy_train, false_acceptance_rate, false_recognition_rate, precision, recall, f1_score = metrics
        if accuracy >= best_accuracy:
            chosen_delta1 = delta1
            best_accuracy = accuracy

        accuracy_list[i] = accuracy
        FAR_list[i] = false_acceptance_rate
        FRR_list[i] = false_recognition_rate
        precision_list[i] = precision
        recall_list[i] = recall
        f1_score_list[i] = f1_score

    performance = {"Delta": delta1_domain_range,
                   "Accuracy": accuracy_list,
                   "False Acceptance Rate": FAR_list,
                   "False Recognition Rate": FRR_list,
                   "Precision": precision_list,
                   "Recall": recall_list,
                   "F1-Score": f1_score_list
                   }

    return chosen_delta1, performance


def _compute_delta1_domain_range(min_distance_to_label_mapping: np.ndarray, tot_num_of_deltas: int) -> np.ndarray:
    min_delta = np.min(min_distance_to_label_mapping[:, 0])
    max_delta = np.max(min_distance_to_label_mapping[:, 0])
    min_delta = float(min_delta)
    max_delta = float(max_delta)

    print("Min delta: ", min_delta, "; Max delta: ", max_delta)
    delta1_domain_range = np.arange(min_delta,
                                    max_delta,
                                    (max_delta - min_delta) / tot_num_of_deltas)
    return delta1_domain_range
