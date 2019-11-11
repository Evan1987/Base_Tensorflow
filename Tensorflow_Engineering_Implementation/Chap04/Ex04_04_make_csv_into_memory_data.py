
"""
Use >>> tf.train.string_input_producer
Use tensorflow queue to read / use data concurrently using more than one process.
One process consume data from cache and do some stuff like training.
Another process read data from disk and save into cache.
"""

import os
import tensorflow as tf
from Tensorflow_Engineering_Implementation.Chap04.utils import CHAP_DATA_PATH
from _utils.utensorflow import get_session_config

train_data_file = os.path.join(CHAP_DATA_PATH, "iris_training.csv")
test_data_file = os.path.join(CHAP_DATA_PATH, "iris_test.csv")
BATCH_SIZE = 32


def create_pipeline(filename: str, batch_size: int, num_epochs: int = None):
    queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    defaults = [[0.], [0.], [0.], [0.], [0.], [0]]
    reader = tf.TextLineReader(skip_header_lines=1)
    _, value = reader.read(queue)
    columns = tf.decode_csv(value, defaults)
    features = tf.stack([col for col in columns[1:-1]])
    label = columns[-1]

    min_after_dequeue = 1000  # The min num of entries in the queue(after dequeue all, there's at least such num left)
    capacity = min_after_dequeue + batch_size

    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue
    )
    return feature_batch, label_batch


if __name__ == '__main__':

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        x_train, y_train = create_pipeline(train_data_file, BATCH_SIZE, 100)  # `num_epochs` is  local variable shit!
        x_test, y_test = create_pipeline(test_data_file, BATCH_SIZE)
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())  # holy shit for local

    with tf.Session(graph=graph, config=get_session_config()) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            i = 0
            while i < 5:
                if coord.should_stop():
                    break
                examples, labels = sess.run([x_train, y_train])
                print(f"Train data: {examples}, Labels: {labels}")
                i += 1
        except tf.errors.OutOfRangeError:
            print("Done reading!")
            examples, labels = sess.run([x_test, y_test])
            print(f"Test data: {examples}, Labels: {labels}")
        except KeyboardInterrupt:
            print("keyboard interruption!")
        finally:
            coord.request_stop()
            coord.join(threads)
