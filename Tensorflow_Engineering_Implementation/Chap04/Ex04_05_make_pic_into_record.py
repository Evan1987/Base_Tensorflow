
"""
Use >>> tf.python_io.TFRecordWriter to save data into `tf.record`
Use >>> tf.TFRecordReader to read from record file.
"""


import os
import tensorflow as tf
from PIL import Image
from Tensorflow_Engineering_Implementation.Chap04.utils import load_man_woman_data, man_woman_path
from Tensorflow_Engineering_Implementation.constant import OUTPUT_HOME
from _utils.utensorflow import get_session_config
from typing import List


output_record = os.path.join(os.path.join(OUTPUT_HOME, "man_woman.record"))
label_alias = {
    "man": 0,
    "woman": 1,
}


def make_record(filenames: List[str], labels: List[str], output_file: str):
    writer = tf.python_io.TFRecordWriter(output_file)
    for file_, label_ in zip(filenames, labels):
        img_bytes = Image.open(file_).resize((256, 256)).tobytes()
        label_ = label_alias[label_]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_])),
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
        writer.write(example.SerializeToString())
    writer.close()


def read_record(record_files: List[str], flag: str = "train"):
    if flag == "train":
        shuffle = True
        batch_size = 3
    else:
        shuffle = False
        batch_size = 1

    filename_queue = tf.train.string_input_producer(record_files, shuffle=shuffle)
    reader = tf.TFRecordReader()

    _, serialized_examples = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_examples,
        features={"label": tf.FixedLenFeature([], tf.int64), "img_raw": tf.FixedLenFeature([], tf.string)})

    image_ = tf.decode_raw(features["img_raw"], tf.uint8)
    image_ = tf.reshape(image_, [256, 256, 3])
    label_ = tf.cast(features["label"], tf.int32)

    if flag == "train":
        image_ = tf.cast(image_, tf.float32) / 255. - 0.5

    img_batch_, label_batch_ = tf.train.batch([image_, label_], batch_size=batch_size, capacity=20)
    return img_batch_, label_batch_


if __name__ == '__main__':
    image_files, image_labels = load_man_woman_data(man_woman_path, is_shuffle=False)
    print(f"Loaded {len(image_files)} images!")

    make_record(image_files, image_labels, output_file=output_record)
    print(f"{output_record} saved done!")

    train_flag = "test"
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        image_batch, label_batch = read_record([output_record], train_flag)
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    with tf.Session(graph=graph, config=get_session_config()) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        dirname = os.path.join(OUTPUT_HOME, f"{train_flag}")
        tf.gfile.MakeDirs(dirname)
        try:
            i = 0
            while i < 5:
                i += 1
                img_list, label_list = sess.run([image_batch, label_batch])
                for img, label in zip(img_list, label_list):
                    target_file = os.path.join(dirname, f"{i}_label_{label}.jpg")
                    Image.fromarray(img, "RGB").save(target_file)
        except tf.errors.OutOfRangeError:
            print("Done!")
        finally:
            coord.request_stop()
            coord.join(threads)
