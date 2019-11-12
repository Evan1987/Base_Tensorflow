
"""
Use >>>tf.data.Dataset api to load data from tf record
"""
import os
import numpy as np
import tensorflow as tf
from _utils.utensorflow import get_session_config
from Tensorflow_Engineering_Implementation.Chap04.utils import CHAP_DATA_PATH, show_images


RECORD_FILE = os.path.join(CHAP_DATA_PATH, "mydata.tfrecords")
IMAGE_SIZE = [256, 256, 3]
BATCH_SIZE = 10
EAGER = True


def read(file: str) -> tf.data.Dataset:
    feature_description = {"label": tf.FixedLenFeature(shape=[], dtype=tf.int64),
                           "img_raw": tf.FixedLenFeature(shape=[], dtype=tf.string)}

    def _parse_one(example_proto):
        parsed_example = tf.parse_single_example(example_proto, feature_description)
        image = tf.decode_raw(parsed_example["img_raw"], out_type=tf.uint8)
        image = tf.reshape(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0 - 0.5

        label = parsed_example["label"]
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, depth=2, on_value=1)
        return image, label

    return tf.data.TFRecordDataset(file).map(_parse_one).batch(BATCH_SIZE).prefetch(BATCH_SIZE)


if __name__ == '__main__':
    if EAGER:  # eager mode
        tf.enable_eager_execution()
        dataset = read(RECORD_FILE)
        for images, labels in dataset:
            images, labels = images.numpy(), labels.numpy()
            images = ((images + 0.5) * 255.).astype(np.uint8)
            show_images(images, labels, n_cols=10)

    else:  # graph mode
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            dataset = read(RECORD_FILE)
            it = dataset.make_one_shot_iterator()
            element = it.get_next()

        with tf.Session(graph=graph, config=get_session_config()) as sess:
            try:
                while True:
                    images, labels = sess.run(element)
                    images = ((images + 0.5) * 255.).astype(np.uint8)
                    show_images(images, labels, n_cols=10)
            except tf.errors.OutOfRangeError:
                print("Reading done!")
