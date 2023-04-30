import glob
import re
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image


def compute_yale_X_and_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack(df['img'].to_numpy())
    y = df['subject'].apply(int).to_numpy()
    return X, y


def compute_yale_face_dataset_from_files(dataset_folder_path: str) -> pd.DataFrame:
    """

    :param dataset_folder_path: path where the Yale Face dataset resides
    :return: pandas dataframe with columns "img", "subject", "A", and "E"
    """

    filelist = glob.glob(f"{dataset_folder_path}/*/*")
    photo_format = re.compile(r"yaleB(\d\d)_P\d\dA([\+\-]\d\d\d)E([\+\-]\d\d).pgm$")

    filelist = [file for file in filelist if photo_format.search(file)]
    data = []
    for filepath in filelist:
        img = np.array(Image.open(filepath))
        img = img.reshape(img.shape[0] * img.shape[1])
        subject, A, E = photo_format.findall(filepath)[0]
        data.append([img, subject, A, E])
    df = pd.DataFrame(data, columns=["img", "subject", "A", "E"])
    return df


