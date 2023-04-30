from typing import Tuple, Union, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import display_html
from IPython.display import display, HTML

from quantum_eigenfaces.linalg import _compute_min_distance_to_index_mapping
from quantum_eigenfaces.utils.utils import _compute_outcomes


def print_dataframes_side_by_side(*dfs: Tuple[pd.DataFrame, ...]):
    stylers = []
    for df in dfs:
        stylers.append(df.style.set_table_attributes("style='display:inline'"))

    html = "".join([styler._repr_html_() for styler in stylers])

    display_html(html, raw=True)


def plot_metrics(metrics: pd.DataFrame, chosen_delta1: float, show: bool = True,
                 file_path: Union[str, None] = None, save: bool = True) -> None:
    delta1s = metrics["Delta"]

    # Set the figure size
    plt.figure(figsize=(15, 5))

    # First graph
    plt.subplot(1, 3, 1)
    plt.plot(delta1s, metrics["Accuracy"], label='Accuracy')
    plt.axvline(x=chosen_delta1, c="red")
    plt.legend()
    plt.xlabel('$\delta$')
    plt.ylabel('Metrics')
    plt.grid(True)

    # Second graph
    f = plt.subplot(1, 3, 2)
    plt.plot(delta1s, metrics["F1-Score"], label='F1 Score')
    plt.plot(delta1s, metrics["Precision"], label='Precision')
    plt.plot(delta1s, metrics["Recall"], label='Recall')
    plt.axvline(x=chosen_delta1, c="red")
    plt.legend()
    plt.xlabel('$\delta$')
    plt.ylabel('Metrics')
    plt.grid(True)

    # Third graph
    plt.subplot(1, 3, 3)
    plt.plot(delta1s, metrics["False Acceptance Rate"], label='False Acceptance Rate')
    plt.plot(delta1s, metrics["False Recognition Rate"], label='False Recognition Rate')
    plt.axvline(x=chosen_delta1, c="red")
    plt.legend()
    plt.xlabel('$\delta$')
    plt.ylabel('Metrics')
    plt.grid(True)
    plt.subplots_adjust(wspace=0.4)

    if show:
        plt.show()

    if save:
        figsize = (6, 8)
        plt.clf()

        plt.figure(figsize=figsize)
        plt.plot(delta1s, metrics["Accuracy"], label='Accuracy')
        plt.axvline(x=chosen_delta1, c="red")
        plt.legend()
        plt.xlabel('$\delta$')
        plt.ylabel('Metrics')
        plt.grid(True)
        filename = 'DELTA_acc.pdf'
        plt.savefig(filename)
        print(f"Saved plot at {filename}")

        plt.clf()
        plt.figure(figsize=figsize)
        plt.plot(delta1s, metrics["F1-Score"], label='F1 Score')
        plt.plot(delta1s, metrics["Precision"], label='Precision')
        plt.plot(delta1s, metrics["Recall"], label='Recall')
        plt.axvline(x=chosen_delta1, c="red")
        plt.legend()
        plt.xlabel('$\delta$')
        plt.ylabel('Metrics')
        plt.grid(True)
        filename = "DELTA_pre_rec_and_f1.pdf"
        plt.savefig(filename)
        print(f"Saved plot at {filename}")

        plt.clf()
        plt.figure(figsize=figsize)
        plt.plot(delta1s, metrics["False Acceptance Rate"], label='False Acceptance Rate')
        plt.plot(delta1s, metrics["False Recognition Rate"], label='False Recognition Rate')
        plt.axvline(x=chosen_delta1, c="red")
        plt.legend()
        plt.xlabel('$\delta$')
        plt.ylabel('Metrics')
        plt.grid(True)
        filename = 'DELTA_far_and_frr.pdf'
        plt.savefig(filename)
        print(f"Saved plot at {filename}")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')


def plot_faces(faces: np.ndarray, matrix_shape: Tuple[int, ...], reshaper: Union[bool, None, Tuple[int, int]] = None):
    """
    print the first matrix_shape[0] * matrix_shape[1] faces you pass in input.

    :param faces: array (num_faces, width, height) or (num_faces, num_flattened_features)
    :param matrix_shape: visualization matrix. one image for each cell of this matrix.
    :param reshaper: False, None or tuple.
    :return:
    """

    r, c = matrix_shape
    gs = gridspec.GridSpec(r, c, top=1, bottom=0., right=3, left=0., hspace=0, wspace=0.5)

    for i, g in enumerate(gs):
        ax = plt.subplot(g)

        if reshaper is not False:
            if reshaper is None:
                face = faces[i]
            else:
                face = faces[i].reshape(reshaper)

        ax.imshow(face, cmap=plt.get_cmap("gray"))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def plot_value_counts(df: pd.DataFrame, column: str):
    """ plot a bar plot with the value counts of the specified column """
    value_count = df[column].value_counts()
    # Plot the value count
    fig, ax = plt.subplots(figsize=(20, 5))  # set the figure size in inches
    ax.bar(value_count.index, value_count.values)
    plt.xlim([0, len(value_count)])
    ax.set_xlabel(column)
    ax.set_ylabel('count')
    ax.set_title(f'Value count of {column}')


def h1(text: str) -> None:
    display(HTML(f"<h1>{text}</h1>"))


def h3(text: str) -> None:
    display(HTML(f"<h3>{text}</h3>"))


BOOLEAN_TO_COLOR_MAPPING = {0: "red", 1: "green", 2: "blue", 3: "orange"}


def plot_imgs_side_by_side(faces: List[np.ndarray],
                           titles: Union[List[str], List[List[str]]],
                           outcomes: Optional[List[int]] = None) -> None:
    """

    :param faces: list of numpy arrays with shape (num imgs, height, width)
    :param titles: list of titles to associate with img on the x-axis.
    :param outcomes: int list of predictions outcomes (0 = false recognition; 1 = true recognition; 2 = true denial)

    0 -> False Recognition
    1 -> True Recognition
    2 -> True Denial
    3 -> False Denial
    """
    print("Legenda")
    display(HTML("<font color=green>Green  = True Recognition </font>"))
    display(HTML("<font color=red>Red    = False Recognition</font>"))
    display(HTML("<font color=blue>Blue   = True Denial</font>"))
    display(HTML("<font color=orange>Orange = False Denial</font>"))

    assert (len(faces) == len(titles)) or (len(faces) == len(titles[0]))
    assert (all([len(face) == len(outcomes) for face in faces]))

    color_titles_with_outcomes = True if outcomes is not None else False

    num_imgs = faces[0].shape[0]
    assert (all(num_imgs == faces[i].shape[0] for i in range(len(faces))))

    ncols = len(faces)
    nrows = faces[0].shape[0]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * 10, 120), dpi=200)
    fig.subplots_adjust(wspace=0.1, hspace=0.5)

    for i in range(nrows):
        for j in range(ncols):

            """
            if type(titles[i] == str):
                title = titles[j]
            else:  # assuming titles is a list of lists.
            """

            title = titles[i][j]
            axs[i, j].set_title(title, fontsize=12)

            if color_titles_with_outcomes is True:
                axs[i, j].title.set_color(BOOLEAN_TO_COLOR_MAPPING[outcomes[i]])

            axs[i, j].imshow(faces[j][i], cmap="gray")
            axs[i, j].axis('off')
            axs[i, j].axis('off')


def plot_comparison_matrix(C_test: np.ndarray,
                           C_train: np.ndarray,
                           x_bar_train: np.ndarray,
                           eigen_faces: np.ndarray,
                           reshaper: Tuple[int, ...],
                           y_test: np.ndarray,
                           y_predict: np.ndarray,
                           y_train: np.ndarray,
                           X_train: np.ndarray,
                           X_test: np.ndarray,
                           epsilon: float
                           ) -> None:
    """
    plot reconstructed validation samples on the left. plot reconstructed most similar img in training set on the right.
    Reconstructed images are plot because the distance comparison is made on the weight vectors, not on the original images.

    if X_train and X_valid are passed, it also plots the original images.
    """

    min_distance_index_mapping = _compute_min_distance_to_index_mapping(C=C_test, C_train=C_train, epsilon=epsilon)

    reconstructed_faces_train = C_train @ eigen_faces + x_bar_train
    reconstructed_faces_train = reconstructed_faces_train.reshape(reconstructed_faces_train.shape[0], *reshaper)
    reconstructed_faces_test = C_test @ eigen_faces + x_bar_train
    reconstructed_faces_test = reconstructed_faces_test.reshape(reconstructed_faces_test.shape[0], *reshaper)

    rec_test_samples = np.zeros_like(reconstructed_faces_test)
    rec_most_similar_imgs_in_train = np.zeros_like(reconstructed_faces_test)
    most_similar_imgs_in_train = np.zeros_like(reconstructed_faces_test)

    for i in range(len(reconstructed_faces_test)):
        rec_test_samples[i] = reconstructed_faces_test[i]

        most_similar_img_index = min_distance_index_mapping.loc[i, "index C training"]

        rec_most_similar_imgs_in_train[i] = reconstructed_faces_train[most_similar_img_index]
        most_similar_imgs_in_train[i] = X_train[most_similar_img_index].reshape(*reshaper)

    outcomes = _compute_outcomes(y_test=y_test, y_predict=y_predict, y_train=y_train)

    test_samples = X_test[0:C_test.shape[0]].reshape(C_test.shape[0], *reshaper)
    faces = [test_samples,
             rec_test_samples,
             rec_most_similar_imgs_in_train,
             most_similar_imgs_in_train]
    titles = [[f"ORI TEST sample (index = {i})",
               f"REC TEST Sample (gt = {y}; pred = {yp})",
               "REC TRAINING NN",
               "ORI TRAINING NN"] for i, (yp, y) in enumerate(zip(y_predict, y_test))]

    plot_imgs_side_by_side(faces=faces, titles=titles, outcomes=outcomes)


def print_eigen_faces_stats(pca, eigen_faces: np.ndarray, n_components: int, reshaper: Union[Tuple[int, int], None]):
    eigen_faces_stats = pd.DataFrame(
        {"Number of Principal Component": [n_components],
         "Shape of Principal Components": [pca.components_.shape],
         "Variance Explained by all Principal Components": [np.sum(pca.explained_variance_ratio_)],
         })

    other_stats = pd.DataFrame({"First 5 Principal Components": list(range(1, len(pca.components_) + 1)),
                                "First 5 PC's Explained Variance Ratio": pca.explained_variance_ratio_,
                                }).head(5)
    print_dataframes_side_by_side(eigen_faces_stats, other_stats)

    if reshaper is not None:
        tmp = " the first 20 " if n_components >= 21 else "all the Eigenfaces"
        h3(f"Have a look at {tmp}:")
        plot_faces(faces=eigen_faces[0:n_components], matrix_shape=(8, int(n_components / 8)), reshaper=reshaper)


def plot_metrics_at_varying_epsilon(results: pd.DataFrame, name: str, epsilon_domain: list[float], logx: bool = False,
                                    tick_step: int = 1):
    epsilons = results["epsilon"].values
    acc = results["Accuracy"].values
    stddev_acc = results["Stddev Accuracy"].values
    stddev_acc_train = results["Stddev Accuracy Train"].values
    stddev_recall = results["Stddev Recall"].values
    stddev_f1 = results["Stddev F1-Score"].values
    stddev_prec = results["Stddev Precision"].values

    acc_train = results["Accuracy Train"].values
    prec = results["Precision"].values
    rec = results["Recall"].values
    f1 = results["F1-Score"].values

    if logx:
        epsilons = [np.log10(epsilon) for epsilon in epsilons]
        epsilon_domain = [np.log10(epsilon) for epsilon in epsilon_domain]
        xlabel = "$\log{\epsilon}}$"
    else:
        xlabel = "$\epsilon$"

    # Set the figure size
    plt.figure(figsize=(10, 10))

    # ------------ First graph -------------
    # plt.subplot(2, 1, 1)

    # Accuracy
    plt.plot(epsilons, acc, label='Accuracy')
    plt.fill_between(epsilons, acc - stddev_acc, acc + stddev_acc, color="lightskyblue")

    # Accuracy Train
    plt.plot(epsilons, acc_train, label='Accuracy Train')
    plt.fill_between(epsilons, acc_train - stddev_acc_train, acc_train + stddev_acc_train, color="wheat")

    # Recall
    plt.plot(epsilons, rec, label='Recall')
    plt.fill_between(epsilons, rec - stddev_recall, rec + stddev_recall, color="mediumaquamarine")

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Metrics')
    plt.xticks(epsilon_domain[0:-1:tick_step])
    plt.title(f'Accuracy, Accuracy Train, and Recall vs. $\epsilon$')
    plt.grid(True)

    file_name = f"{name}_epsilon_wrt_acc_train_and_recall.pdf"
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Saved plot at {file_name}")
    plt.show()
    plt.clf()

    # ------------ Second Graph -------------

    plt.figure(figsize=(10, 10))

    # plt.subplot(2, 1, 2)
    # F1 Score
    plt.plot(epsilons, f1, label='F1 Score')
    plt.fill_between(epsilons, f1 - stddev_f1, f1 + stddev_f1, color="lightskyblue")

    # Precision
    plt.plot(epsilons, prec, label='Precision')
    plt.fill_between(epsilons, prec - stddev_prec, prec + stddev_prec, color="wheat")

    # Recall
    plt.plot(epsilons, rec, label='Recall')
    plt.fill_between(epsilons, rec - stddev_recall, rec + stddev_recall, color="mediumaquamarine")

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('Metrics')
    plt.xticks(epsilon_domain[0:-1:tick_step])

    plt.title(f'F1 Score, Precision, and Recall vs. $\epsilon$')

    plt.grid(True)
    file_name = f"{name}_epsilon_wrt_acc_train_and_recall.pdf"
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Saved plot at {file_name}")

