import itertools
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class QuantumRuntimeParameters:
    m: int
    k: int
    sqrt_p: float
    best_mu: float
    max_c_train_norm: float

    def describe(self):
        print("Quantum Runtime Parameters")
        print("--------------------------")
        print(f"best_mu:          {self.best_mu}")
        print(f"max_c_train_norm: {self.max_c_train_norm}")
        print(f"m:                {self.m}")
        print(f"k:                {self.k}")
        print(f"sqrt_p:           {self.sqrt_p}")


@dataclass
class TrainingConfig:
    n_components: int


@dataclass
class TuningConfig:
    tot_num_of_deltas: int


@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def describe(self):
        print(f"train shape: {self.X_train.shape}")
        print(f"valid shape: {self.X_valid.shape}")
        print(f"test shape : {self.X_test.shape}")


def recover_X_and_y(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = _recover_X(dataset=dataset)
    y = np.array(dataset["y"], dtype="int")
    return X, y


def _recover_X(dataset: pd.DataFrame) -> np.ndarray:
    num_features = len(dataset["X"][0])
    num_rows = dataset.shape[0]
    X = np.zeros((num_rows, num_features))
    for i in range(num_rows):
        X[i, :] = np.array(dataset["X"][i], dtype="int")
    return X


def create_dataset(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"X": data.tolist(), "y": labels})


def keep_rows_based_on_condition(df: pd.DataFrame, mask: Dict[str, List[Any]]) -> pd.DataFrame:
    # WRONG
    a = df.copy()
    for column, values in mask.items():
        a = a[a[column].isin(values)]

    return a


def get_list_of_all_possible_combinations(a, b) -> List[Tuple[Any]]:

    subsets_a = list(itertools.chain.from_iterable(itertools.combinations(a, r) for r in range(1, len(a) + 1)))
    subsets_b = list(itertools.chain.from_iterable(itertools.combinations(b, r) for r in range(1, len(b) + 1)))

    # TODO: don't judge me, I implemented this very badly.

    product = [(x, y) for x in tqdm(subsets_a) for y in subsets_b]
    return product


def _compute_outcomes(y_test: np.ndarray, y_predict: np.ndarray, y_train: np.ndarray) -> List[int]:
    """
    0 -> False Recognition
    1 -> True Recognition
    2 -> True Denial
    3 -> False Denial

    :param y_test:
    :param y_predict:
    :param y_train:
    :return:
    """
    outcomes = []
    for i in range(len(y_test)):

        if y_test[i] in y_train:
            if y_test[i] == y_predict[i]:
                outcomes.append(1)
            elif y_predict[i] > 0:
                outcomes.append(0)
            else:
                # prediction is "undefined element in same category" or undefined element in another category£
                outcomes.append(3)  # --> False Denial

        else:
            if y_predict[i] > 0:
                outcomes.append(0)

            else:
                outcomes.append(2)

    return outcomes


def compute_balance(y_array: np.ndarray) -> float:
    """ check how well distributed subjects are """
    subjects_in_array = set(y_array)
    num_occ_of_subject = np.zeros((max(subjects_in_array) + 1,))
    for y in y_array:
        num_occ_of_subject[y] += 1
    num_occ_of_subject = np.array([occ for i, occ in enumerate(num_occ_of_subject) if i in subjects_in_array])
    expected_count = len(y_array)/len(subjects_in_array)
    balances = num_occ_of_subject / expected_count
    balance = np.mean(balances)
    return balance


def compute_test_anomalies_percentage(y_test, y_train):
    subjects_in_test_but_not_in_training = set(y_test) - set(y_train)
    anomalies_percentage = sum([y in subjects_in_test_but_not_in_training for y in y_test]) / len(y_test)
    return anomalies_percentage


def compute_test_subjects_in_training(y_test, y_train):
    subjects_in_training = set(y_train)
    in_training_percentage = sum([y in subjects_in_training for y in y_test]) / len(y_test)
    return in_training_percentage


def compute_split_statistics(y_train: np.ndarray, y_valid: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:

    train_balance = compute_balance(y_train)
    valid_balance = compute_balance(y_valid)
    test_balance = compute_balance(y_test)
    test_anomalies_percentage = compute_test_anomalies_percentage(y_test=y_test,
                                                                  y_train=y_train)
    test_subjects_in_training_percentage = compute_test_subjects_in_training(y_test=y_test,
                                                                             y_train=y_train)

    num_train_samples = y_train.shape[0]
    num_valid_samples = y_valid.shape[0]
    num_test_samples = y_test.shape[0]

    num_train_subjects = len(set(y_train))
    num_valid_subjects = len(set(y_valid))
    num_test_subjects = len(set(y_test))

    split_statistics = pd.DataFrame({"Num Train Samples": [num_train_samples],
                                     "Num Valid Samples": [num_valid_samples],
                                     "Num Test Samples": [num_test_samples],
                                     "Num Train subjects": [num_train_subjects],
                                     "Num Valid Subjects": [num_valid_subjects],
                                     "Num Test Subjects": [num_test_subjects],
                                     "Train Balance": [train_balance],
                                     "Valid Balance": [valid_balance],
                                     "Test Balance": [test_balance],
                                     "Test Anomalies %": [test_anomalies_percentage],
                                     "Test Subjects in Training Set %": [test_subjects_in_training_percentage]})
    return split_statistics
