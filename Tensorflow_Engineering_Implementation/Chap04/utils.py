
import os
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import Tuple, List, Union, Optional
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


def show_images(images: np.ndarray, labels: np.ndarray, n_cols: int = 8,
                image_shape: Optional[Union[Tuple, List]] = None):
    """
    Show data in a plot.
    :param images: multi pictures' grey level data.
    :param labels: the pictures' corresponding labels.
    :param n_cols: the num of columns in a figure
    :param image_shape: 1-d tuple or list of single image's shape
    """
    n_rows = math.ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows))
    for i, (image, label) in enumerate(zip(images, labels)):
        r = i // n_cols
        c = i % n_cols
        if image_shape:
            image = np.reshape(image, image_shape)
        axe: plt.Axes = axes[r, c] if n_rows > 1 else axes[c]
        axe.imshow(np.squeeze(image))
        axe.axis("off")
        axe.set_title(str(label))
    fig.show()
