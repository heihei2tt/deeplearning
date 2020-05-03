"""
TFRECORD内部使用了Protocal Buffer二进制数据编码方案，只占用一个内存块。只需一次性加载一个二进制文件的方式即可，简单快速，尤其对大型训练数据很友好。
而且当数据量大时，可制作多个TFRECORD文件，提高处理效率。
"""

import tensorflow as tf
import numpy as np
import cv2
import os

data_dir = "./data"
train_set_name = "train.tfrecords"
val_set_name = "val.tfrecords"

# tf_example模块
"""
message Example{
    Features features = 1;
}
message Features{
    map<string, Feature> feature = 1;
};
message Feature{
    oneof kind{
        # 三种数据形式
        ByteList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
"""


def write_image_to_tfrecords():
    # TFRcord生成器
    writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, train_set_name))

    # fpath_list = os.listdir(os.path.join(data_dir, 'train'))
    # 当目录中存在多级文件夹时，使用os.walk
    for root, dirs, fnames in os.walk("dataset/train"):
        # 遍历每一个图像
        for fname in fnames:
            # 图像数据
            """
            cv2.imread(fpath, CV_LOAD_IMAGE_ANYDEPTH)或者cv2.imread(fpath, -1) 保持原深度
            cv2.imread(fpath, CV_LOAD_IMAGE_COLOR)或者1，强制转换成彩色图像，即3通道
            cv2.imread(fpath, CV_LOAD_IMAGE_GRAYSCALE)或者0，强制转换成灰度图像，即1通道
            """
            img_data = cv2.imread("dataset/train" + os.sep + dirs + os.sep(fname))
            # label
            img_label = dirs
            tf_example = tf.train.Example(features=tf.train.features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_label])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_data.tobytes()]))
                }
            ))
            writer.write(tf_example.SerializeToString())
    print("Done training_set writing.")
    writer.close()


# 从tfrecords文件中读一个batch的数据
def read_batch_image(file_queue, batch_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    # ******************
    # 解析单个样本，但是怎么传给tf.train.shuffle_batch呢？
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'image_raw': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [128, 128, 3])
    label = tf.decode_raw(features['label'], tf.uint8)
    # ******************
    # 要搞清楚这几个参数的含义
    image_batch, label_batch = tf.train.shuffle_batch(tensors=[image, label],
                                                      batch_size=batch_size,
                                                      capacity=2000,
                                                      min_after_dequeue=4000)
    return image_batch, label_batch


# def train():
#     img_fname_queue = tf.train.string_input_producer(string_tensor=tf.train.match_filenames_once("train.tfrecords"),
#                                                      num_epochs=10,
#                                                      shuffle=True)




































