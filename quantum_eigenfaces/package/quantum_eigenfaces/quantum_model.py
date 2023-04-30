from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from IPython.display import display, HTML

from sklearn.decomposition import PCA

from quantum_eigenfaces.linalg import compute_weight_vectors
from quantum_eigenfaces.testing import iterative_compute_metrics_and_y_predict
from quantum_eigenfaces.tuning import tuning
from quantum_eigenfaces.utils.runtime import compute_best_mu, compute_quantum_runtime
from quantum_eigenfaces.utils.utils import QuantumRuntimeParameters, compute_split_statistics
from quantum_eigenfaces.utils.visualization import print_eigen_faces_stats, plot_comparison_matrix, plot_metrics

np.random.seed(1234)


class QuantumModel:
    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_valid: np.ndarray,
                 y_valid: np.ndarray,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 reshaper: Optional[Tuple[int, int]],
                 tot_num_of_deltas: int = 800,
                 visuals: bool = False):

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.reshaper = reshaper

        self.chosen_delta1 = None
        self.tuning_performance = None
        self.y_predict = None
        self.epsilon = None

        self.tot_num_of_deltas = tot_num_of_deltas
        self.classification_tqdm = None
        self.visuals = visuals

        self.use_norm_threshold = None
        self.norm_ratios_valid = None
        self.norm_ratios_test = None
        self.norm_threshold = None

        self.quantum_runtime_parameters = None

    def fit(self, n_components: int, delta1: Optional[float] = None, norm_threshold: Optional[float] = None) -> Optional[pd.DataFrame]:

        self.norm_threshold = norm_threshold

        if self.visuals is True:
            print("********* DATASET STATS **************")
            self._show_dataset_stats()

        print("********* TRAINING **************")
        self._training(n_components=n_components)

        print("********* TUNING **************")
        if delta1 is None:
            performance = self._tuning()
            return performance
        else:
            print("delta1 has been chosen by the user. No delta1 tuning is performed.")
            self.chosen_delta1 = delta1

    def runtime(self, epsilon: float, num_components: int, verbose: bool = False) -> tuple[np.ndarray, float]:
        """ compute quantum and classical runtime on the test set """

        # compute quantum runtime parameters
        m = self.X_train.shape[1]
        k = num_components
        p = self.C_train.shape[0]
        sqrt_p = np.sqrt(p)
        best_mu = compute_best_mu(self.eigen_faces)[1]
        max_c_train_norm = np.max(np.linalg.norm(self.C_train, axis=1))

        self.quantum_runtime_parameters = QuantumRuntimeParameters(m=m,
                                                                   k=k,
                                                                   sqrt_p=sqrt_p,
                                                                   best_mu=best_mu,
                                                                   max_c_train_norm=max_c_train_norm)

        if verbose:
            self.quantum_runtime_parameters.describe()

        # Compute quantum runtime
        U_bar = self.X_test - self.x_bar_train
        qrt = compute_quantum_runtime(sqrt_p=sqrt_p,
                                      mu=best_mu,
                                      max_c_train_norm=max_c_train_norm,
                                      U_bar=U_bar,
                                      epsilon=epsilon)

        # Compute classical run time
        crt = (m * k) + (p * k)

        return qrt, crt

    def predict(self, epsilon: float, use_norm_threshold: bool, num_iterations: int = 100, xi: float = 0.01):
        self.use_norm_threshold = use_norm_threshold
        if not self.use_norm_threshold:
            print("The norm threshold is not going to be used.")

        self.epsilon = epsilon
        print("********* TESTING **************")

        metrics, stddevs, y_predict = iterative_compute_metrics_and_y_predict(X_test=self.X_test,
                                                                              y_test=self.y_test,
                                                                              C_test=self.C_test,
                                                                              y_train=self.y_train,
                                                                              C_train=self.C_train,
                                                                              delta1=self.chosen_delta1,
                                                                              epsilon=epsilon,
                                                                              num_iterations=num_iterations,
                                                                              use_norm_threshold=self.use_norm_threshold,
                                                                              norm_threshold=self.norm_threshold,
                                                                              norm_ratios_test=self.norm_ratios_test,
                                                                              xi=xi
                                                                              )

        if self.visuals:
            self._show_performance(metrics=metrics, stddevs=stddevs, y_predict=y_predict, C_test=self.C_test)
        return metrics, stddevs

    def run_all(self, n_components: int, epsilon: float):

        self.fit(n_components=n_components)
        metrics, stddevs = self.predict(epsilon=epsilon, use_norm_threshold=self.use_norm_threshold)

        return metrics, stddevs

    # ---------------- HIGH-LEVEL SUB-FUNCTIONS -------------------

    def _show_dataset_stats(self):
        if self.reshaper:
            print(f"Dataset Dimensionality: {self.X_train.shape[1]} ({self.reshaper[0]}, {self.reshaper[1]})")
        else:
            print(f"Dataset Dimensionality: {self.X_train.shape[1]}")
        split_statistics = compute_split_statistics(y_train=self.y_train, y_valid=self.y_valid, y_test=self.y_test)
        display(HTML(split_statistics.to_html()))

    def _training(self, n_components: int) -> None:

        self.x_bar_train = np.mean(self.X_train, axis=0)
        dataset = self.X_train - self.x_bar_train
        self.pca = PCA(n_components=n_components)
        self.pca.fit(dataset)
        self.eigen_faces = self.pca.components_
        self.C_train = compute_weight_vectors(X=self.X_train,
                                              x_bar=self.x_bar_train,
                                              V=self.eigen_faces)

        # Compute norm threshold
        norm_x_train = np.linalg.norm(self.X_train - self.x_bar_train, axis=1)
        norm_c_train = np.linalg.norm(self.C_train, axis=1)
        if self.norm_threshold is None:
            self.norm_threshold = round(min(norm_c_train / norm_x_train), 6)
            print(f"Normalization Threshold is: {self.norm_threshold}")
        else:
            print(f"Normalization Threshold has been chosen by the user: {self.norm_threshold}")

        # Compute valid/test weight vectors
        self.C_valid = compute_weight_vectors(X=self.X_valid,
                                              x_bar=self.x_bar_train,
                                              V=self.eigen_faces)
        self.C_test = compute_weight_vectors(X=self.X_test,
                                             x_bar=self.x_bar_train,
                                             V=self.eigen_faces)

        # compute valid reconstruction ratios
        norm_x_valid = np.linalg.norm(self.X_valid - self.x_bar_train, axis=1)
        norm_c_valid = np.linalg.norm(self.C_valid, axis=1)
        self.norm_ratios_valid = norm_c_valid / norm_x_valid

        # compute test reconstruction ratios
        norm_x_test = np.linalg.norm(self.X_test - self.x_bar_train, axis=1)
        norm_c_test = np.linalg.norm(self.C_test, axis=1)
        self.norm_ratios_test = norm_c_test / norm_x_test

        if self.visuals is True and self.reshaper is not None:
            print_eigen_faces_stats(pca=self.pca,
                                    eigen_faces=self.eigen_faces,
                                    n_components=n_components,
                                    reshaper=self.reshaper)

    def _show_performance(self, metrics: Tuple[float, ...], stddevs: Tuple[float, ...], y_predict: List[int],
                          C_test: np.ndarray) -> None:
        assert C_test is not None, "the _predict method should have been called first."

        accuracy, accuracy_train, false_acceptance_rate, false_recognition_rate, precision, recall, f1_score, tnr = metrics

        test_performance = pd.DataFrame({"AVG Accuracy": [accuracy],
                                         "AVG Accuracy train": [accuracy_train],
                                         "AVG False Acceptance Rate": [false_acceptance_rate],
                                         "AVG False Recognition Rate": [false_recognition_rate],
                                         "AVG Precision": [precision],
                                         "AVG Recall": [recall],
                                         "AVG F1 score": [f1_score],
                                         "AVG True Negative Rate": [tnr]})

        display(HTML(test_performance.to_html()))

        stddev_accuracy, stddev_accuracy_train, stddev_false_acceptance_rate, stddev_false_recognition_rate, stddev_precision, stddev_recall, stddev_f1_score, stddev_tnr = stddevs

        test_stddev_performance = pd.DataFrame({"STDDEV Accuracy": [stddev_accuracy],
                                                "STDDEV Accuracy train": [stddev_accuracy_train],
                                                "STDDEV False Acceptance Rate": [stddev_false_acceptance_rate],
                                                "STDDEV False Recognition Rate": [stddev_false_recognition_rate],
                                                "STDDEV Precision": [stddev_precision],
                                                "STDDEV Recall": [stddev_recall],
                                                "STDDEV F1 score": [stddev_f1_score],
                                                "STDDEV True Negative Rate": [stddev_tnr]})

        display(HTML(test_stddev_performance.to_html()))

        if self.visuals is True and self.reshaper is not None:
            plot_comparison_matrix(C_test=C_test,
                                   y_test=self.y_test,
                                   C_train=self.C_train,
                                   x_bar_train=self.x_bar_train,
                                   eigen_faces=self.eigen_faces,
                                   reshaper=self.reshaper,
                                   y_predict=y_predict,
                                   y_train=self.y_train,
                                   X_train=self.X_train,
                                   X_test=self.X_test,
                                   epsilon=self.epsilon)

    # ---------------- TUNING -------------------

    def _tuning(self, show_plot: bool = True) -> pd.DataFrame:

        chosen_delta1, performance = tuning(y_train=self.y_train,
                                            C_train=self.C_train,
                                            X_valid=self.X_valid,
                                            y_valid=self.y_valid,
                                            C_valid=self.C_valid,
                                            tot_num_of_deltas=self.tot_num_of_deltas,
                                            use_norm_threshold=self.use_norm_threshold,
                                            norm_threshold=self.norm_threshold,
                                            norm_ratios_valid=self.norm_ratios_valid)
        print(f"Chosen delta1: {chosen_delta1}")
        self.chosen_delta1 = chosen_delta1
        tuning_performance = pd.DataFrame(performance)
        if show_plot:
            plot_metrics(metrics=tuning_performance,
                         chosen_delta1=self.chosen_delta1)

        if self.visuals is True:
            display(HTML(tuning_performance.sort_values(by="Accuracy", ascending=False).head().to_html()))

        return tuning_performance
