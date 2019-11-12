
"""
Using >>>tf.data.Dataset api to transform image data into `dataset`
"""

import numpy as np
import tensorflow as tf
from _utils.utensorflow import get_session_config
from Tensorflow_Engineering_Implementation.Chap04.utils import man_woman_path, load_man_woman_data, show_images


SIZE = [96, 96]
CHANNEL = 1
BATCH_SIZE = 10


class ImageTransformer(object):
    def __init__(self, shuffle: bool = False, crop: bool = False, brightness: bool = False,
                 contrast: bool = False, norm: bool = False, flatten: bool = False, rotated: bool = False):
        self.shuffle = shuffle
        self.crop = crop
        self.brightness = brightness
        self.contrast = contrast
        self.norm = norm
        self.flatten = flatten
        self.rotated = rotated

    def __call__(self, image: tf.Tensor):
        image = tf.image.random_flip_left_right(image)  # randomly left-right flip
        image = tf.image.random_flip_up_down(image)  # randomly up-down flip

        if self.crop:
            s = np.random.randint(low=int(SIZE[0] * 0.8), high=SIZE[0], size=2)
            image = tf.image.random_crop(image, size=[s[0], s[1], CHANNEL])

        if self.brightness:  # randomly change brightness
            image = tf.image.random_brightness(image, max_delta=4)

        if self.contrast:  # randomly change contrast
            image = tf.image.random_contrast(image, lower=0.2, upper=1.4)

        if self.shuffle:
            image = tf.random_shuffle(image)  # shuffle the 0-axis

        image = tf.image.resize(image, SIZE)

        if self.rotated:
            from tensorflow.contrib.image import rotate
            image = rotate(image, 30)

        if self.norm:
            image = image / 255.0

        if self.flatten:
            image = tf.reshape(image, [SIZE[0] * SIZE[1] * CHANNEL])
        return image


def parse_one(image_file: str, image_label: str):
    image = tf.read_file(image_file)
    image: tf.Tensor = tf.image.decode_image(image)
    image.set_shape([None, None, None])

    transformer = ImageTransformer(shuffle=False, crop=False, brightness=False, contrast=False, norm=False, rotated=True)
    image = transformer(image)
    return image, image_label


if __name__ == '__main__':
    image_files, image_labels = load_man_woman_data(man_woman_path)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        dataset = tf.data.Dataset.from_tensor_slices((image_files, image_labels))\
            .map(parse_one, num_parallel_calls=1)\
            .batch(BATCH_SIZE)
        it = dataset.make_one_shot_iterator()
        element = it.get_next()

    with tf.Session(graph=graph, config=get_session_config()) as sess:
        try:
            while True:
                image_batch, label_batch = sess.run(element)
                show_images(image_batch, label_batch, n_cols=8)

        except tf.errors.OutOfRangeError:
            print("Read done!")




