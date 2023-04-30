import numpy as np
import scipy


def compute_weight_vectors(X: np.ndarray, x_bar: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Convert each data point into a linear combination of principal components

    Legenda:
    n = points
    m = features
    k = principal components

    :param X: data (n, m)
    :param x_bar (m,)
    :param V: principal components matrix (k, m)
    :return: array (n, k)
    """
    n_components = V.shape[0]
    n = len(X)
    C = np.zeros((n, n_components))
    for i in range(0, n):
        C[i, :] = compute_weight_vector(x=X[i], x_bar=x_bar, V=V)
    return C


def compute_weight_vector(x: np.ndarray, x_bar: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Convert each data point into a linear combination of principal components

    Legenda:
    m = features
    k = principal components

    :param x: data (m,)
    :param x_bar: mean vector (m,)
    :param V: principal components matrix (k, m)
    :return: array (k,)
    """
    c = V @ (x - x_bar)
    return c


def _compute_min_distance_to_label_mapping(C: np.ndarray,
                                           C_train: np.ndarray,
                                           y_train: np.ndarray,
                                           epsilon: float = 0) -> np.ndarray:

    min_squared_distance_to_label_mapping = np.zeros((C.shape[0], 2))

    squared_distance_matrix = scipy.spatial.distance.cdist(C, C_train) ** 2
    shape = squared_distance_matrix.shape
    noisy_squared_distance_matrix = squared_distance_matrix + np.random.uniform(-epsilon, epsilon, shape)
    noisy_squared_distance_matrix = noisy_squared_distance_matrix.clip(min=0)

    min_squared_distance_to_label_mapping[:, 0] = np.min(noisy_squared_distance_matrix, axis=1)

    min_distances = min_squared_distance_to_label_mapping[:, 0]

    n_test = noisy_squared_distance_matrix.shape[0]

    # compute argmin
    for i in range(n_test):

        min_indices = np.where(noisy_squared_distance_matrix[i] == min_distances[i])[0]

        num_min_indices = len(min_indices)
        training_corresponding_random_index = min_indices[np.random.randint(0, num_min_indices)]
        del min_indices
        min_squared_distance_to_label_mapping[i, 1] = y_train[training_corresponding_random_index]

    return min_squared_distance_to_label_mapping


def _compute_min_distance_to_index_mapping(C: np.ndarray,
                                           C_train: np.ndarray,
                                           epsilon: float = 0) -> np.ndarray:

    min_squared_distance_to_index_mapping = np.zeros((C.shape[0], 2))

    squared_distance_matrix = scipy.spatial.distance.cdist(C, C_train) ** 2
    shape = squared_distance_matrix.shape
    noisy_squared_distance_matrix = squared_distance_matrix + np.random.uniform(-epsilon, epsilon, shape)
    noisy_squared_distance_matrix = noisy_squared_distance_matrix.clip(min=0)

    min_squared_distance_to_index_mapping[:, 0] = np.min(noisy_squared_distance_matrix, axis=1)

    min_distances = min_squared_distance_to_index_mapping[:, 0]

    n_test = noisy_squared_distance_matrix.shape[0]

    # compute argmin
    for i in range(n_test):

        min_indices = np.where(noisy_squared_distance_matrix[i] == min_distances[i])[0]

        num_min_indices = len(min_indices)
        training_corresponding_random_index = min_indices[np.random.randint(0, num_min_indices)]
        del min_indices
        min_squared_distance_to_index_mapping[i, 1] = training_corresponding_random_index

    return min_squared_distance_to_index_mapping
