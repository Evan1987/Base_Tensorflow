
"""
Use >>> tf.train.slice_input_producer
Use tensorflow queue to read / use data concurrently using more than one process.
One process consume data from cache and do some stuff like training.
Another process read data from disk and save into cache.
"""

import tensorflow as tf
from _utils.utensorflow import get_session_config
from Tensorflow_Engineering_Implementation.Chap04.utils import load_mnist_data, show_images, mnist_data_path, BATCH_SIZE
from typing import List


def get_batches(image_files: List[str], image_labels: List[str], width: int, height: int, channels: int, batch_size: int):
    queue = tf.train.slice_input_producer([image_files, image_labels])
    image_, label_ = queue
    image_ = tf.read_file(image_)
    image_ = tf.image.decode_bmp(image_, channels=channels)
    image_ = tf.image.resize_image_with_crop_or_pad(image_, target_height=height, target_width=width)
    image_ = tf.image.per_image_standardization(image_)
    image_batch_, label_batch_ = tf.train.batch([image_, label_], batch_size=batch_size, num_threads=16)
    image_batch_ = tf.cast(image_batch_, dtype=tf.float32)
    label_batch_ = tf.reshape(label_batch_, shape=[batch_size])
    return image_batch_, label_batch_


if __name__ == '__main__':
    images, labels = load_mnist_data(mnist_data_path)
    print(f"Loaded {len(images)} image files.")

    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        image_batch, label_batch = get_batches(images, labels, 28, 28, 1, BATCH_SIZE)
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph, config=get_session_config()) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in range(5):
                if coord.should_stop():
                    break
                image_list, label_list = sess.run([image_batch, label_batch])
                show_images(image_list, label_list, image_shape=(28, 28))
        except tf.errors.OutOfRangeError:
            print("Done!")
        finally:
            coord.request_stop()
            coord.join(threads)
