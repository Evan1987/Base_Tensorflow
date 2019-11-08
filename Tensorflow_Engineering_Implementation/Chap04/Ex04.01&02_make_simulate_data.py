

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

N_EPOCHS = 20


def gen_data(batch_size: int = 100):
    x = np.linspace(-1, 1, batch_size)
    y = 2 * x + np.random.rand(*x.shape) * 0.3
    yield x, y


def gen_data2(n_epochs: int, batch_size: int = 100):
    for _ in range(n_epochs):
        x = np.linspace(-1, 1, batch_size)
        y = 2 * x + np.random.rand(*x.shape) * 0.3
        yield shuffle(x, y)


graph = tf.Graph()
with graph.as_default():
    input_x = tf.placeholder(dtype=tf.float32, shape=None)
    input_y = tf.placeholder(dtype=tf.float32, shape=None)


if __name__ == '__main__':
    sess = tf.Session(graph=graph)
    for xv, yv in gen_data():
        x, y = sess.run([input_x, input_y], feed_dict={input_x: xv, input_y: yv})

    x, y = list(gen_data())[0]
    plt.plot(x, y, "ro", label="original data")
    plt.legend()
    plt.show()
