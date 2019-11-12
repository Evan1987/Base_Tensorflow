

import tensorflow as tf
from typing import List
from _utils.utensorflow import get_session_config


train_data = [1, 2, 3, 4, 5]
test_data = [10, 20, 30, 40, 50]


def make_dataset(data: List[int]):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    it = dataset.make_one_shot_iterator()
    return it


if __name__ == '__main__':
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        train_it = make_dataset(train_data)
        test_it = make_dataset(test_data)

        handle = tf.placeholder(dtype=tf.string, shape=[])

        # handle the output by the handle(string value)
        iterator = tf.data.Iterator.from_string_handle(handle, output_types=train_it.output_types)
        element = iterator.get_next()

    with tf.Session(graph=graph, config=get_session_config()) as sess:
        train_handle = sess.run(train_it.string_handle())
        test_handle = sess.run(test_it.string_handle())
        print(f"Train string handle: {train_handle}")
        print(f"Test string handle: {test_handle}")

        print(sess.run(element, feed_dict={handle: train_handle}))
        print(sess.run(element, feed_dict={handle: test_handle}))
