

import os
import glob
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from Tensorflow_Engineering_Implementation.constant import SOURCE_HOME
from typing import Tuple, List

LABELS = list(range(10))
BATCH_SIZE = 16
mnist_data_path = os.path.join(SOURCE_HOME, "Chap04/mnist_digits_images")


def load_data(data_dir: str) -> Tuple[List[str], List[int]]:
    labels = []
    files = []
    for file in tqdm(glob.glob(os.path.join(data_dir, "*/*.bmp"))):
        label = os.path.dirname(os.path.relpath(file, start=data_dir))
        labels.append(int(label))
        files.append(file)
    return shuffle(np.asarray(files), np.asarray(labels))


def get_queue(images: List[str], labels: List[int], input_width: int, input_height: int, channels: int):

    def process(images, labels):
        labels = tf.cast(labels, dtype=tf.int32)
        images = tf.read_file(images)
        images = tf.image.decode_bmp(image_files, channels)
        images = tf.image.resize_image_with_crop_or_pad(images, input_height, input_width)
        images = tf.image.per_image_standardization(images)
        images = tf.cast(images, dtype=tf.float64)
        return images, labels

    queue = tf.data.Dataset.from_tensor_slices((images, labels)).map(process)
    return queue


def show_images(images, labels):
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


if __name__ == '__main__':
    image_files, image_labels = load_data(mnist_data_path)
    print(f"Loaded {len(image_files)} image files.")

    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        queue = get_queue(image_files, image_labels, 28, 28, 1)
        init = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        for step in range(10):
            if coord.should_stop():
                break
            image_list, label_list = sess.run(queue.batch(BATCH_SIZE))

            show_images(image_list, label_list)
    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()
        coord.join(threads)
