
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import Tuple, List
from Tensorflow_Engineering_Implementation.constant import SOURCE_HOME


LABELS = list(range(10))
BATCH_SIZE = 16
CHAP_DATA_PATH = os.path.join(SOURCE_HOME, "Chap04")
mnist_data_path = os.path.join(CHAP_DATA_PATH, "mnist_digits_images")
man_woman_path = os.path.join(CHAP_DATA_PATH, "man_woman")


class _PicLoader(object):
    """Load pic data from disk."""
    def __init__(self, pattern: str = "*/*.jpg"):
        self.pattern = pattern

    def __call__(self, data_dir: str, is_shuffle: bool = True):
        """:returns: (1) the src path of picture. (2) the label of picture."""
        labels = []
        files = []
        for file in tqdm(glob.glob(os.path.join(data_dir, self.pattern))):
            label = os.path.dirname(os.path.relpath(file, start=data_dir))
            labels.append(label)
            files.append(file)

        if is_shuffle:
            return shuffle(np.asarray(files), np.asarray(labels))
        return np.asarray(files), np.asarray(labels)


def load_mnist_data(data_dir: str, is_shuffle: bool = True) -> Tuple[List[str], List[str]]:
    loader = _PicLoader("*/*.bmp")
    return loader(data_dir, is_shuffle)


def load_man_woman_data(data_dir: str, is_shuffle: bool = False) -> Tuple[List[str], List[str]]:
    loader = _PicLoader("*/*.jpg")
    return loader(data_dir, is_shuffle)


def show_images(images: np.ndarray, labels: np.ndarray):
    """
    Show data in a plot.
    :param images: multi pictures' grey level data.
    :param labels: the pictures' corresponding labels.
    """
    n_cols = 8
    n_rows = math.ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows))
    for i, (image, label) in enumerate(zip(images, labels)):
        r = i // n_cols
        c = i % n_cols
        axe: plt.Axes = axes[r, c]
        axe.imshow(np.reshape(image, (28, 28)))
        axe.axis("off")
        axe.set_title(str(label))
    fig.show()
