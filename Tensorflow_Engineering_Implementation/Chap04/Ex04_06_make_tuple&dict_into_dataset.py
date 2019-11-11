

"""
Use >>>tf.data.Dataset api to transform tuples and dicts
"""

import numpy as np
import tensorflow as tf


batch_size = 10


def generate_data(n: int = 100):
    x_ = np.linspace(-1, 1, n)
    y_ = 2 * x_ + np.random.randn(*x_.shape) * 0.3
    return x_, y_


def get_dataset(x: np.ndarray, y: np.ndarray, output_format: str = "tuple"):
    if output_format == "tuple":
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        map_func = lambda d: (d[0], tf.cast(d[1], tf.int32))
    elif output_format == "dict":
        dataset = tf.data.Dataset.from_tensor_slices({"x": x, "y": y})
        map_func = lambda d: (d["x"], tf.cast(d["y"], tf.int32))
    else:
        raise ValueError(f"Invalid dataset {output_format}.")

    dataset = dataset.shuffle(100).repeat(3).map(map_func, num_parallel_calls=1)
    return dataset

