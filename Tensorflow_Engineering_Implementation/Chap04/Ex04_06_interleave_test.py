
"""
A test script for testing `interleave` method
"""

import os
import tensorflow as tf
from Tensorflow_Engineering_Implementation.Chap04.utils import CHAP_DATA_PATH
from _utils.utensorflow import get_session_config


def parse_fn(line: str):
    print(line)
    return line


if __name__ == '__main__':
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        data = tf.data.Dataset.list_files(os.path.join(CHAP_DATA_PATH, "testset/*.txt"), shuffle=False)\
            .interleave(
                map_func=lambda file: tf.data.TextLineDataset(file).map(parse_fn, num_parallel_calls=1),
                cycle_length=2,
                block_length=2)
        it = data.make_one_shot_iterator()
        element = it.get_next()

    with tf.Session(graph=graph, config=get_session_config()) as sess:
        try:
            while True:
                lines = sess.run(element)
                print(lines)
        except tf.errors.OutOfRangeError:
            print("Reading done!")
