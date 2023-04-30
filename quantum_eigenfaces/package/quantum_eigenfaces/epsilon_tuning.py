from typing import Optional

import numpy as np
import pandas as pd

from quantum_eigenfaces.quantum_model import QuantumModel
from quantum_eigenfaces.utils.utils import DataSplit, TrainingConfig, TuningConfig


def epsilon_tuning(datasplit: DataSplit,
                   training_config: TrainingConfig,
                   tuning_config: TuningConfig,
                   reshaper,
                   epsilons,
                   use_norm_threshold: bool,
                   num_iterations: int = 100,
                   delta1: float = None,
                   xi: float = 0.01,
                   visuals: bool = False,
                   norm_threshold: Optional[float] = None,
                   ds_name: str = "",
                   compute_runtimes: bool = False):

    X_train = datasplit.X_train
    y_train = datasplit.y_train
    X_valid = datasplit.X_valid
    y_valid = datasplit.y_valid
    X_test = datasplit.X_test
    y_test = datasplit.y_test
    n_components = training_config.n_components
    tot_num_of_deltas = tuning_config.tot_num_of_deltas

    AVG_ACCS, AVG_ACCS_TR, AVG_FARs, AVG_FRRs, AVG_PREs, AVG_RECs, AVG_F1s = (np.zeros(len(epsilons)) for _ in range(7))
    STDDEV_ACCS, STDDEV_ACCS_TR, STDDEV_FARs, STDDEV_FRRs, STDDEV_PREs, STDDEV_RECs, STDDEV_F1s = (np.zeros(len(epsilons)) for _ in range(7))

    qm = QuantumModel(X_train=X_train,
                      y_train=y_train,
                      X_valid=X_valid,
                      y_valid=y_valid,
                      X_test=X_test,
                      y_test=y_test,
                      reshaper=reshaper,
                      tot_num_of_deltas=tot_num_of_deltas,
                      visuals=visuals)

    qm.fit(n_components=n_components, delta1=delta1, norm_threshold=norm_threshold)

    avg_rts = []
    stddev_rts = []
    crts = []

    for i, epsilon in enumerate(epsilons):

        print("_______")
        print(f"Epsilon: {epsilon}")

        if compute_runtimes:
            print("Computing runtimes...")
            qrt, crt = qm.runtime(epsilon=epsilon, num_components=n_components)
            avg_rts.append(np.mean(qrt))
            stddev_rts.append(np.std(qrt))
            del qrt
            crts.append(crt)

        print("Compute predictions...")
        metrics, stddevs = qm.predict(epsilon=epsilon,
                                      use_norm_threshold=use_norm_threshold,
                                      num_iterations=num_iterations,
                                      xi=xi)
        avg_accuracy, avg_acc_tr, avg_far, avg_frr, avg_pre, avg_rec, avg_f1s = metrics
        AVG_ACCS[i] = avg_accuracy
        AVG_ACCS_TR[i] = avg_acc_tr
        AVG_FARs[i] = avg_far
        AVG_FRRs[i] = avg_frr
        AVG_PREs[i] = avg_pre
        AVG_RECs[i] = avg_rec
        AVG_F1s[i] = avg_f1s

        stddev_acc, stddev_acc_tr, stddev_far, stddev_frr, stddev_pre, stddev_rec, stddev_f1s = stddevs

        STDDEV_ACCS[i] = stddev_acc
        STDDEV_ACCS_TR[i] = stddev_acc_tr
        STDDEV_FARs[i] = stddev_far
        STDDEV_FRRs[i] = stddev_frr
        STDDEV_PREs[i] = stddev_pre
        STDDEV_RECs[i] = stddev_rec
        STDDEV_F1s[i] = stddev_f1s

        summary = pd.DataFrame({"epsilon": epsilons,
                                "Accuracy": AVG_ACCS,
                                "Accuracy Train": AVG_ACCS_TR,
                                "False Acceptance Rate": AVG_FARs,
                                "False Recognition Rate": AVG_FRRs,
                                "Precision": AVG_PREs,
                                "Recall": AVG_RECs,
                                "F1-Score": AVG_F1s,
                                "Stddev Accuracy": STDDEV_ACCS,
                                "Stddev Accuracy Train": STDDEV_ACCS_TR,
                                "Stddev False Acceptance Rate": STDDEV_FARs,
                                "Stddev False Recognition Rate": STDDEV_FRRs,
                                "Stddev Precision": STDDEV_PREs,
                                "Stddev Recall": STDDEV_RECs,
                                "Stddev F1-Score": STDDEV_F1s})

        summary.to_csv("tmp.csv")

    if compute_runtimes:

        rt_df = pd.DataFrame({"epsilon": epsilons,
                              "classical runtime": crts,
                              "average runtime": avg_rts,
                              "stddev runtime": stddev_rts})

    summary = pd.DataFrame({"epsilon": epsilons,
                            "Accuracy": AVG_ACCS,
                            "Accuracy Train": AVG_ACCS_TR,
                            "False Acceptance Rate": AVG_FARs,
                            "False Recognition Rate": AVG_FRRs,
                            "Precision": AVG_PREs,
                            "Recall": AVG_RECs,
                            "F1-Score": AVG_F1s,
                            "Stddev Accuracy": STDDEV_ACCS,
                            "Stddev Accuracy Train": STDDEV_ACCS_TR,
                            "Stddev False Acceptance Rate": STDDEV_FARs,
                            "Stddev False Recognition Rate": STDDEV_FRRs,
                            "Stddev Precision": STDDEV_PREs,
                            "Stddev Recall": STDDEV_RECs,
                            "Stddev F1-Score": STDDEV_F1s})

    if compute_runtimes:
        return summary, rt_df

    return summary


if __name__ == "__main__":
    pass
