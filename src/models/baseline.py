from typing import Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import Config


def get_random_predictions(
    test_ds: tf.data.Dataset, class_labels: list
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(Config.SEED)

    y_true = []
    y_pred = []

    for _, true_labels in tqdm(test_ds):
        for label in true_labels:
            y_random_pred = np.random.randint(low=0, high=len(class_labels))

            y_true.append(int(label.numpy()))
            y_pred.append(y_random_pred)

    return y_true, y_pred


def get_most_common_predictions(
    test_ds: tf.data.Dataset, most_common_country_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []

    for _, true_labels in tqdm(test_ds):
        for label in true_labels:
            y_true.append(int(label.numpy()))
            y_pred.append(most_common_country_index)

    return y_true, y_pred
