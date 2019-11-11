
"""
A test script for testing multi ops of `dataset`
"""

import tensorflow as tf
from typing import Callable

test_cases = {}


def register(name: str):
    def wrapper(function: Callable):
        test_cases[name] = function
        return function
    return wrapper


@register("range")
def range_dataset():
    return tf.data.Dataset.range(5)  # [0, 1, 2, 3, 4]


@register("zip")
def zip_dataset():
    d1 = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    d2 = tf.data.Dataset.from_tensor_slices([2, 3, 4, 5, 6])
    return tf.data.Dataset.zip((d1, d2))  # [(1, 2), (2, 3), (3, 4),...]


@register("concat")
def concat_dataset():
    d1 = tf.data.Dataset.from_tensor_slices([-1, -2, -3, -4, -5])
    d2 = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    return d1.concatenate(d2)


@register("repeat")
def repeat_dataset():
    d = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    return d.repeat(2)


@register("shuffle")
def shuffle_dataset():
    d = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    return d.shuffle(10)


@register("batch")
def batch_dataset():
    d = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    return d.batch(3)


@register("padded")
def padded_batch_dataset():
    d = tf.data.Dataset.from_tensor_slices([[1, 2], [2, 3], [3, 4]])
    return d.padded_batch(batch_size=2, padded_shapes=(4,))  # [[1, 2, 0, 0], [2, 3, 0, 0]], [[3, 4, 0, 0]]


@register("flat_map")
def flat_map_dataset():
    d = tf.data.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
    return d.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))  # [1, 2, 3, 4, 5, 6]


@register("interleave")
def interleave_dataset():
    d = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    return d.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(3), cycle_length=2, block_length=2)


@register("filter")
def filter_dataset():
    d = tf.data.Dataset.from_tensor_slices([[0.0, 0.0], [3.0, 4.0]])
    return d.filter(lambda x: tf.reduce_sum(x) > 0)  # [3., 4.]


@register("apply")
def apply_dataset():
    """Split data by `windows_size`, and in each window odd row and even row into two groups by `key_func`.
    Foreach group, return group values in batches(batch_size=10) by `reduce_func`
    [ 0  2  4  6  8 10 12 14 16 18]  - window 0, group 0, batch 0
    [20 22 24 26 28 30 32 34 36 38]  - window 0, group 0, batch 1
    [ 1  3  5  7  9 11 13 15 17 19]  - window 0, group 1, batch 0
    [21 23 25 27 29 31 33 35 37 39]  - window 0, group 1, batch 1
    [40 42 44 46 48]                 - window 1, group 0, batch 0
    [41 43 45 47 49]                 - window 1, group 1, batch 0
    """
    from tensorflow.data.experimental import group_by_window
    d = tf.data.Dataset.range(50)
    return d.apply(group_by_window(key_func=lambda x: x % 2, reduce_func=lambda _, els: els.batch(10), window_size=20))


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    for key, func in test_cases.items():
        print("*" * 10 + f" {key} testing " + "*" * 10)
        data = func()
        it = data.make_one_shot_iterator()
        element = it.get_next()
        try:
            while True:
                print(sess.run(element))
        except tf.errors.OutOfRangeError:
            print("Reading Done!")
        input("Press Enter to continue...")

