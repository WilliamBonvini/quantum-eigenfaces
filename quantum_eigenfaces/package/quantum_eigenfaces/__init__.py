import os
package_path = os.path.dirname(os.path.realpath(__file__))
resources_path = os.path.join(package_path, "resources")
datasets_path = os.path.join(resources_path, "dataset")
face_datasets_path = os.path.join(datasets_path, "face_recognition")

# YALE FACE DATASET
CROPPED_YALE_DATASET_PATH = os.path.join(face_datasets_path, "CroppedYale")
yale_processed_path = os.path.join(datasets_path, "face_recognition", "processed")
TRAIN_YALE_DATA_PATH = os.path.join(yale_processed_path, "train.parquet.gzip")
VALID_YALE_DATA_PATH = os.path.join(yale_processed_path, "valid.parquet.gzip")
TEST_YALE_DATA_PATH = os.path.join(yale_processed_path, "test.parquet.gzip")
IMGS_PATH = os.path.join(resources_path, "imgs")
VANERIO_SPLITS_PATH = os.path.join(datasets_path, "face_recognition", "processed", "vanerio")

# LFWCROP DATASET
LFWCROP_DATASET_PATH = os.path.join(face_datasets_path, "lfwcrop_grey")
LFWCROP_TRAIN_PATH = os.path.join(LFWCROP_DATASET_PATH, "processed", "train.parquet.gzip")
LFWCROP_VALID_PATH = os.path.join(LFWCROP_DATASET_PATH, "processed", "valid.parquet.gzip")
LFWCROP_TEST_PATH = os.path.join(LFWCROP_DATASET_PATH, "processed", "test.parquet.gzip")

# ORL
_orl_base_path = os.path.join(face_datasets_path, "ORL")
ORL_DATASET_PATH = os.path.join(_orl_base_path, "original")
ORL_TRAIN_PATH = os.path.join(_orl_base_path, "processed", "train.parquet.gzip")
ORL_VALID_PATH = os.path.join(_orl_base_path, "processed", "valid.parquet.gzip")
ORL_TEST_PATH = os.path.join(_orl_base_path, "processed", "test.parquet.gzip")

# RESULTS
MNIST_NORM_CHECK_AND_VARYING_EPSILON_PATH = os.path.join(resources_path,
                                                         "results",
                                                         "MNIST",
                                                         "norm_check_and_varying_epsilon.csv")

MNIST_NO_NORM_CHECK_AND_NO_EPSILON_PATH = os.path.join(resources_path,
                                                       "results",
                                                       "MNIST",
                                                       "no_norm_check_and_no_varying_epsilon.csv")

MNIST_NO_NORM_CHECK_AND_VARYING_EPSILON_PATH = os.path.join(resources_path,
                                                            "results",
                                                            "MNIST",
                                                            "no_norm_check_and_varying_epsilon.csv")


MNIST_QUANTUM_DELTA_TUNED_NORM_THRESHOLD_FALSE_ANOMALIES_TRUE = os.path.join(resources_path,
                                                                            "results",
                                                                            "MNIST",
                                                                            "quantum-delta_tuned-norm_threshold_false-anomalies_true.csv")
