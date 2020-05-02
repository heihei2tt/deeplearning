# -*- coding:utf-8 -*-
"""
#====#====#====#====
# Project Name:     U-net
# File Name:        unet-TF-withBatchNormal
# Date:             2/17/18 8:18 PM
# Using IDE:        PyCharm Community Edition
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin
# BlogPage:         http://blog.csdn.net/qq_30239975
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#====
"""
import tensorflow as tf
import argparse
import os
import cv2
import numpy as np

TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.tfrecords'

# ORIGIN_PREDICT_DIRECTORY = 'oss://xixiyuanqu/mv_data_set/test'
ORIGIN_PREDICT_DIRECTORY = 'mv_data_set/test'
INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 112, 112, 8
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 112, 112, 1

EPOCH_NUM = 40

TRAIN_BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 8

TEST_SET_SIZE = 1000  # 这个数据集暂时没有，1000是假设的
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1

# PREDICT_SAVED_DIRECTORY = 'oss://xixiyuanqu/mv_data_set/predictions'
PREDICT_SAVED_DIRECTORY = "mv_data_set/predictions/"
EPS = 10e-5
FLAGS = None
# 共9类，结球生菜，7个小品种，全部others地物都作为背景，背景用0表示
CLASS_NUM = 9
# 全部训练样本的个数
# TRAIN_NUM = 13142
# CHECK_POINT_PATH = 'oss://xixiyuanqu/mv_data_set/saved_models/model.ckpt'
CHECK_POINT_PATH = 'mv_data_set/saved_models/model.ckpt'


def write_tfrecord(writer, filelist, train_set_size):
    for index, file_name in enumerate(filelist):
        # rgb图像patch的路径
        rgb_path = os.path.join('./mv_data_set/train', file_name)
        # 还要找到对应的多光谱图像patch
        # cls_idx: 类别号，rgb_idx：第多少幅遥感图像，rgb_or_mv_idx：从0到5，0表示rgb，1～5为多光谱
        # patch_idx：随机crop得到的样本的标号
        cls_idx, rgb_idx, rgb_or_mv_idx, patch_idx = file_name.split('-')
        blue_path = os.path.join('./mv_data_set/train', cls_idx+'-'+rgb_idx+'-'+str(1)+'-'+patch_idx)
        green_path = os.path.join('./mv_data_set/train', cls_idx+'-'+rgb_idx+'-'+str(2)+'-'+patch_idx)
        red_path = os.path.join('./mv_data_set/train', cls_idx+'-'+rgb_idx+'-'+str(3)+'-'+patch_idx)
        nir_path = os.path.join('./mv_data_set/train', cls_idx+'-'+rgb_idx+'-'+str(4)+'-'+patch_idx)
        rededge_path = os.path.join('./mv_data_set/train', cls_idx+'-'+rgb_idx+'-'+str(5)+'-'+patch_idx)
        # 读数据
        rgb_data = cv2.imread(rgb_path)
        b_rgb_data, g_rgb_data, r_rgb_data = cv2.split(rgb_data)
        blue_data = cv2.imread(blue_path, -1)
        green_data = cv2.imread(green_path, -1)
        red_data = cv2.imread(red_path, -1)
        nir_data = cv2.imread(nir_path, -1)
        rededge_data = cv2.imread(rededge_path, -1)
        # 将8个通道的数据堆叠到一起
        sample_data = cv2.merge([b_rgb_data, g_rgb_data, r_rgb_data, blue_data, green_data, red_data, nir_data, rededge_data])
        train_image = cv2.resize(src=sample_data, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
        train_image = np.asarray(a=train_image, dtype=np.uint8)

        # 要找到对应的label图像
        label_file_name = cls_idx + '-' + rgb_idx + '-' + patch_idx
        label_file_path = os.path.join('./mv_data_set/label', label_file_name)
        # label image shape is (512, 512), its channel = 1
        label_data = cv2.imread(label_file_path, -1)
        label_data = cv2.resize(src=label_data, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
        # 第0类的二值label为[0,1], 第一类的二值label为[0,2]，以此类推，第8类的二值label为[0,8]，其中0表示背景
        # file_name 的形式为0-123.png，
        # 第0类为others地物，全部当作背景，因此label=0
        # if label_file_name.startswith('0'):
        if cls_idx == '0':
            label_data[:, :] = 0
        else:
            # 其它8类为：结球生菜+7个小品种，第1类为jieqiu，第2类为kuju，。。。，第8类为xibanya
            # label_dict = {"others": 0, "jieqiu": 1, "kuju": 2, "luomanni": 3, "luoshahong": 4, "luoshalv": 5,
            # "lvluo": 6, "naiyoulv": 7, "xibanya": 8}
            # 样本图像命名为：1-3666, 2-3333, 类似
            label_data[label_data <= 100] = 0
            label_data[label_data > 100] = int(cls_idx)
        print "max min pixel scales ", np.amax(label_data), np.amin(label_data)

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_data.tobytes()])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_image.tobytes()]))
        }))
        writer.write(example.SerializeToString())
        if (index + 1) == train_set_size:
            # print index
            # print ("Done training_set writing %.2f%%" % (index / train_set_size * 100))
            print ("Done training_set writing 100%")
    writer.close()


def write_img_to_tfrecords():
    import cv2
    import numpy as np
    filename_list = os.listdir(os.path.join(FLAGS.data_dir, 'train'))

    # 找出所有的rgb图像，
    rgb_list = []
    for filename in filename_list:
        # 例如，2-6-0-215.png表示第2类的第6个遥感图像生成的rgb的第215个样本，对应的多光谱patch的命名为
        # 2-6-1-215.png, 2-6-2-215.png, 2-6-3-215.png, 2-6-4-215.png, 2-6-5-215.png
        # cls_idx, rgb_idx, rgb_or_mv_idx, patch_idx = filename.split('-')[2]
        if filename.split('-')[2] == '0':
            rgb_list.append(filename)
    print("number of all samples", len(rgb_list))

    train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, TRAIN_SET_NAME))  # 要生成的文件
    validation_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, VALIDATION_SET_NAME))

    np.random.shuffle(rgb_list)
    # 总的样本数量
    samples_cnt = len(rgb_list)
    # 取80%作为训练样本
    train_set_size = int(0.9 * samples_cnt)
    # 剩余的10%作为验证样本
    validation_set_size = int(0.1 * samples_cnt)

    train_path = rgb_list[:train_set_size]
    validation_path = rgb_list[train_set_size:]

    write_tfrecord(train_set_writer, train_path, train_set_size)
    write_tfrecord(validation_set_writer, validation_path, validation_set_size)


def read_check_tfrecords():
    import cv2
    train_file_path = os.path.join(FLAGS.data_dir, TRAIN_SET_NAME)
    train_image_filename_queue = tf.train.string_input_producer(
        string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=1, shuffle=True)
    train_images, train_labels = read_image(train_image_filename_queue)
    # one_hot_labels = tf.to_float(tf.one_hot(indices=train_labels, depth=CLASS_NUM))
    with tf.Session() as sess:  # 开始一个会话
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        example, label = sess.run([train_images, train_labels])
        cv2.imshow('image', example)
        cv2.imshow('label', label * 255)
        cv2.waitKey(0)
        # print(sess.run(one_hot_labels))
        coord.request_stop()
        coord.join(threads)
    print("Done reading and checking")


def read_image(file_queue):
    reader = tf.TFRecordReader()
    # key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
            })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # print('image ' + str(image))
    print "image tensor", image
    image = tf.reshape(image, [INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL])
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image = tf.image.resize_images(image, (IMG_HEIGHT, IMG_WIDE))
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label = tf.decode_raw(features['label'], tf.uint8)
    print "label tensor", label
    # label = tf.cast(label, tf.int64)
    label = tf.reshape(label, [OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
    # label = tf.decode_raw(features['image_raw'], tf.uint8)
    # print(label)
    # label = tf.reshape(label, shape=[1, 4])
    return image, label


def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    min_after_dequeue = 2000
    capacity = 4000
    # image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
    image_batch, label_batch = tf.train.shuffle_batch(
        tensors=[img, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    # one_hot_labels = tf.to_float(tf.one_hot(indices=label_batch, depth=CLASS_NUM))
    one_hot_labels = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE])
    return image_batch, one_hot_labels


class Unet:
    def __init__(self):
        print('New U-net Network')
        self.input_image = None
        self.input_label = None
        # self.train_num = 0
        # self.cast_image = None
        # self.cast_label = None
        self.keep_prob = None
        self.lamb = None
        self.result_expand = None
        self.is_training = None
        self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contract_layer = {}
        self.w = {}
        self.b = {}

    def init_w(self, shape, name):
        with tf.name_scope('init_w'):
            stddev = tf.sqrt(x=2.0 / (shape[0] * shape[1] * shape[2]))
            # stddev = 0.01p
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32), name=name)
            tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
            return w

    # @staticmethod
    def init_b(shape, name):
        with tf.name_scope('init_b'):
            return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

    # @staticmethod
    def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
        from tensorflow.python.training.moving_averages import assign_moving_average

        with tf.variable_scope(name):
            params_shape = x.shape[-1:]
            print "params_shape = ", params_shape
            moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
            moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)

            def mean_var_with_update():
                mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
                with tf.control_dependencies([
                    assign_moving_average(moving_mean, mean_this_batch, decay),
                    assign_moving_average(moving_var, variance_this_batch, decay)
                ]):
                    return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
            if affine:  # 如果要用beta和gamma进行放缩
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=eps)
            else:
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,  variance_epsilon=eps)
            return normed

    # @staticmethod
    def copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling):
        result_from_contract_layer_shape = tf.shape(result_from_contract_layer)
        result_from_upsampling_shape = tf.shape(result_from_upsampling)
        result_from_contract_layer_crop = \
            tf.slice(
                input_=result_from_contract_layer,
                begin=[
                    0,
                    (result_from_contract_layer_shape[1] - result_from_upsampling_shape[1]) // 2,
                    (result_from_contract_layer_shape[2] - result_from_upsampling_shape[2]) // 2,
                    0
                ],
                size=[
                    result_from_upsampling_shape[0],
                    result_from_upsampling_shape[1],
                    result_from_upsampling_shape[2],
                    result_from_upsampling_shape[3]
                ]
            )
        # result_from_contract_layer_crop = result_from_contract_layer
        return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=3)

    @staticmethod
    def conv(w, bias,):
        return

    def set_up_unet(self, batch_size):
        # input
        with tf.name_scope('input'):
            # learning_rate = tf.train.exponential_decay()
            self.input_image = tf.placeholder(
                dtype=tf.float32, shape=[batch_size, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL], name='input_images')
            # self.cast_image = tf.reshape(
            # 	tensor=self.input_image,
            # 	shape=[batch_size, INPUT_IMG_WIDE, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL]
            # )

            # For softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
            # using one-hot
            # self.input_label = tf.placeholder(
            # 	dtype=tf.uint8, shape=[OUTPUT_IMG_WIDE, OUTPUT_IMG_WIDE], name='input_labels'
            # )
            # self.cast_label = tf.reshape(
            # 	tensor=self.input_label,
            # 	shape=[batch_size, OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT]
            # )

            # For sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
            # not using one-hot coding
            self.input_label = tf.placeholder(
                dtype=tf.int32, shape=[batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE], name='input_labels'
            )
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            normed_batch = self.batch_norm(x=self.input_image, is_training=self.is_training, name='input')

        # layer 1
        with tf.name_scope('layer_1'):
            # conv_1
            self.w[1] = self.init_w(shape=[3, 3, INPUT_IMG_CHANNEL, 64], name='w_1')
            self.b[1] = self.init_b(shape=[64], name='b_1')
            result_conv_1 = tf.nn.conv2d(
                input=normed_batch, filter=self.w[1], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[1])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_1_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch, name='relu_1')

            # conv_2
            self.w[2] = self.init_w(shape=[3, 3, 64, 64], name='w_2')
            self.b[2] = self.init_b(shape=[64], name='b_2')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[2], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[2])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_1_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contract_layer[1] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
            print "layer 1, result_maxpool ", result_maxpool
            # print "layer 1, result_dropout ", result_dropout

        # layer 2
        with tf.name_scope('layer_2'):
            # conv_1
            self.w[3] = self.init_w(shape=[3, 3, 64, 128], name='w_3')
            self.b[3] = self.init_b(shape=[128], name='b_3')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[3], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[3])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_2_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[4] = self.init_w(shape=[3, 3, 128, 128], name='w_4')
            self.b[4] = self.init_b(shape=[128], name='b_4')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[4], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[4])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_2_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contract_layer[2] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
            print "layer 2, result_maxpool ", result_maxpool
            # print "layer 2, result_dropout ", result_dropout

        # layer 3
        with tf.name_scope('layer_3'):
            # conv_1
            self.w[5] = self.init_w(shape=[3, 3, 128, 256], name='w_5')
            self.b[5] = self.init_b(shape=[256], name='b_5')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[5], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[5])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_3_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[6] = self.init_w(shape=[3, 3, 256, 256], name='w_6')
            self.b[6] = self.init_b(shape=[256], name='b_6')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[6], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[6])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_3_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contract_layer[3] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
            print "layer 3, result_maxpool ", result_maxpool
            # print "layer 3, result_dropout ", result_dropout

        # layer 4
        with tf.name_scope('layer_4'):
            # conv_1
            self.w[7] = self.init_w(shape=[3, 3, 256, 512], name='w_7')
            self.b[7] = self.init_b(shape=[512], name='b_7')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[7], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[7])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_4_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[8] = self.init_w(shape=[3, 3, 512, 512], name='w_8')
            self.b[8] = self.init_b(shape=[512], name='b_8')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[8], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[8])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_4_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')
            self.result_from_contract_layer[4] = result_relu_2  # 该层结果临时保存, 供上采样使用

            # maxpool
            result_maxpool = tf.nn.max_pool(
                value=result_relu_2, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
            print "layer 4, result_maxpool ", result_maxpool
            # print "layer 4, result_dropout ", result_dropout

        # layer 5 (bottom)
        with tf.name_scope('layer_5'):
            # conv_1
            self.w[9] = self.init_w(shape=[3, 3, 512, 1024], name='w_9')
            self.b[9] = self.init_b(shape=[1024], name='b_9')
            result_conv_1 = tf.nn.conv2d(
                input=result_dropout, filter=self.w[9], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[9])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_5_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[10] = self.init_w(shape=[3, 3, 1024, 1024], name='w_10')
            self.b[10] = self.init_b(shape=[1024], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[10], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[10])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_5_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')

            # up sample
            """
            conv2d_transpose(input, filter, output_shape, strides, padding='VALID', name=None)
            filter: [filter_height, filter_width, <big> out_channel, in_channel </big>]
            output_shape: a 1-D Tensor or a N-D shape list, e.g., [batch, out_height, out_width, out_channel]
            """
            self.w[11] = self.init_w(shape=[2, 2, 512, 1024], name='w_11')
            self.b[11] = self.init_b(shape=[512], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[11],
                # output_shape must be pre-set.
                output_shape=[batch_size, 14, 14, 512],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up_b = tf.nn.bias_add(result_up, self.b[11])
            normed_batch = self.batch_norm(x=result_up_b, is_training=self.is_training, name='layer_5_conv_up')
            result_relu_3 = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
            print "layer 5, normed_batch ", normed_batch
            # print "layer 5, result_dropout ", result_dropout

        # layer 6
        with tf.name_scope('layer_6'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[4], result_from_upsampling=result_dropout)
            # print(result_merge)

            # conv_1
            self.w[12] = self.init_w(shape=[3, 3, 1024, 512], name='w_12')
            self.b[12] = self.init_b(shape=[512], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[12], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[12])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_6_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[13] = self.init_w(shape=[3, 3, 512, 512], name='w_10')
            self.b[13] = self.init_b(shape=[512], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[13], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[13])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_6_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')
            # print(result_relu_2.shape[1])

            # up sample
            self.w[14] = self.init_w(shape=[2, 2, 256, 512], name='w_11')
            self.b[14] = self.init_b(shape=[256], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[14],
                output_shape=[batch_size, 28, 28, 256],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up_b = tf.nn.bias_add(result_up, self.b[14])
            normed_batch = self.batch_norm(x=result_up_b, is_training=self.is_training, name='layer_6_conv_up')
            result_relu_3 = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
            print "layer 6, normed_batch ", normed_batch
            # print "layer 6, result_dropout ", result_dropout

        # layer 7
        with tf.name_scope('layer_7'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[3], result_from_upsampling=result_dropout)

            # conv_1
            self.w[15] = self.init_w(shape=[3, 3, 512, 256], name='w_12')
            self.b[15] = self.init_b(shape=[256], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[15], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[15])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_7_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[16] = self.init_w(shape=[3, 3, 256, 256], name='w_10')
            self.b[16] = self.init_b(shape=[256], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[16], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[16])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_7_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')

            # up sample
            self.w[17] = self.init_w(shape=[2, 2, 128, 256], name='w_11')
            self.b[17] = self.init_b(shape=[128], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[17],
                output_shape=[batch_size, 56, 56, 128],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up_b = tf.nn.bias_add(result_up, self.b[17])
            normed_batch = self.batch_norm(x=result_up, is_training=self.is_training, name='layer_7_up')
            result_relu_3 = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
            print "layer 7, normed_batch ", normed_batch
            # print "layer 7, result_dropout ", result_dropout

        # layer 8
        with tf.name_scope('layer_8'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[2], result_from_upsampling=result_dropout)

            # conv_1
            self.w[18] = self.init_w(shape=[3, 3, 256, 128], name='w_12')
            self.b[18] = self.init_b(shape=[128], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[18], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[18])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_8_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[19] = self.init_w(shape=[3, 3, 128, 128], name='w_10')
            self.b[19] = self.init_b(shape=[128], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[19], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[19])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_8_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')

            # up sample
            self.w[20] = self.init_w(shape=[2, 2, 64, 128], name='w_11')
            self.b[20] = self.init_b(shape=[64], name='b_11')
            result_up = tf.nn.conv2d_transpose(
                value=result_relu_2, filter=self.w[20],
                output_shape=[batch_size, 112, 112, 64],
                strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
            result_up_b = tf.nn.bias_add(result_up, self.b[20])
            normed_batch = self.batch_norm(x=result_up_b, is_training=self.is_training, name='layer_8_up')
            result_relu_3 = tf.nn.relu(features=normed_batch, name='relu_3')

            # dropout
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
            print "layer 8, normed_batch ", normed_batch
            # print "layer 8, result_dropout ", result_dropout

        # layer 9
        with tf.name_scope('layer_9'):
            # copy, crop and merge
            result_merge = self.copy_and_crop_and_merge(
                result_from_contract_layer=self.result_from_contract_layer[1], result_from_upsampling=result_dropout)

            # conv_1
            self.w[21] = self.init_w(shape=[3, 3, 128, 64], name='w_12')
            self.b[21] = self.init_b(shape=[64], name='b_12')
            result_conv_1 = tf.nn.conv2d(
                input=result_merge, filter=self.w[21], strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b[21])
            normed_batch = self.batch_norm(x=result_conv_b_1, is_training=self.is_training, name='layer_9_conv_1')
            result_relu_1 = tf.nn.relu(features=normed_batch, name='relu_1')

            # conv_2
            self.w[22] = self.init_w(shape=[3, 3, 64, 64], name='w_10')
            self.b[22] = self.init_b(shape=[64], name='b_10')
            result_conv_2 = tf.nn.conv2d(
                input=result_relu_1, filter=self.w[22], strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b[22])
            normed_batch = self.batch_norm(x=result_conv_b_2, is_training=self.is_training, name='layer_9_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu_2')

            # convolution to [batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
            self.w[23] = self.init_w(shape=[1, 1, 64, CLASS_NUM], name='w_11')
            self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
            result_conv_3 = tf.nn.conv2d(
                input=result_relu_2, filter=self.w[23],
                strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
            result_conv_b_3 = tf.nn.bias_add(result_conv_3, self.b[23])
            normed_batch = self.batch_norm(x=result_conv_b_3, is_training=self.is_training, name='layer_9_conv_3')
            # self.prediction = tf.nn.relu(tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='relu_3')
            # self.prediction = tf.nn.sigmoid(x=tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='sigmoid_1')
            print "the final feature maps: ", normed_batch
            self.prediction = normed_batch

        # softmax loss
        with tf.name_scope('softmax_loss'):
            # using one-hot
            # self.loss = \
            # 	tf.nn.softmax_cross_entropy_with_logits(labels=self.cast_label, logits=self.prediction, name='loss')

            # not using one-hot
            self.loss = \
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        # accuracy
        with tf.name_scope('accuracy'):
            # using one-hot
            # self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.cast_label, axis=3))

            # not using one-hot
            self.correct_prediction = \
                tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        # Gradient Descent
        with tf.name_scope('Gradient_Descent'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

    def train(self):
        train_file_path = os.path.join(FLAGS.data_dir, TRAIN_SET_NAME)
        train_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM, shuffle=True)
        ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        train_images, train_labels = read_image_batch(train_image_filename_queue, TRAIN_BATCH_SIZE)
        print "train_labels", train_labels
        tf.summary.scalar("loss", self.loss_mean)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                while not coord.should_stop():
                    # total_batch = int(TRAIN_NUM / TRAIN_BATCH_SIZE)
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    # for i in range(total_batch):
                    example, label = sess.run([train_images, train_labels])  # 在会话中取出image和label
                    # print(label)
                    lo, acc, summary_str = sess.run(
                        [self.loss_mean, self.accuracy, merged_summary],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 0.7,
                            self.lamb: 0.004, self.is_training: True}
                    )
                    summary_writer.add_summary(summary_str, epoch)
                    # print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    # if epoch % 1 == 0:
                    print('epoch %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    sess.run(
                        [self.train_step],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 0.7,
                            self.lamb: 0.004, self.is_training: True}
                    )
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                all_parameters_saver.save(sess=sess, save_path=ckpt_path)
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print("Done training")

    def validate(self):
        validation_file_path = os.path.join(FLAGS.data_dir, VALIDATION_SET_NAME)
        validation_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(validation_file_path), num_epochs=1, shuffle=True)
        ckpt_path = CHECK_POINT_PATH
        validation_images, validation_labels = read_image_batch(validation_image_filename_queue, VALIDATION_BATCH_SIZE)
        # tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                while not coord.should_stop():
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    example, label = sess.run([validation_images, validation_labels])  # 在会话中取出image和label
                    # print(label)
                    lo, acc = sess.run(
                        [self.loss_mean, self.accuracy],
                        feed_dict={
                            self.input_image: example, self.input_label: label, self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_training: False}
                    )
                    # summary_writer.add_summary(summary_str, epoch)
                    # print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    # if epoch % 1 == 0:
                    print('epoch %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('Done validating -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print('Done validating')

    def test(self):
        import cv2
        import numpy as np
        # test_file_path = os.path.join(FLAGS.data_dir, TEST_SET_NAME)
        test_file_path = os.path.join(FLAGS.data_dir, VALIDATION_SET_NAME)
        test_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(test_file_path), num_epochs=1, shuffle=True)
        ckpt_path = CHECK_POINT_PATH
        test_images, test_labels = read_image_batch(test_image_filename_queue, TEST_BATCH_SIZE)
        # tf.summary.scalar("loss", self.loss_mean)
        # tf.summary.scalar('accuracy', self.accuracy)
        # merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sum_acc = 0.0
            try:
                epoch = 0
                while not coord.should_stop():
                    # Run training steps or whatever
                    # print('epoch ' + str(epoch))
                    example, label = sess.run([test_images, test_labels])  # 在会话中取出image和label
                    # print(label)
                    image, acc = sess.run(
                        [tf.argmax(input=self.prediction, axis=3), self.accuracy],
                        feed_dict={
                            self.input_image: example, self.input_label: label,
                            self.keep_prob: 1.0, self.lamb: 0.004, self.is_training: False
                        }
                    )
                    # # [0 1]
                    # print "values of image", np.unique(image[0])
                    # # <type 'numpy.ndarray'>
                    # print type(image[0])
                    # # int64
                    # print image.dtype
                    # # (512, 512)
                    # print image[0].shape
                    sum_acc += acc
                    epoch += 1
                    # image has values: [0,   1,   2,   3,   4,   5,   6,   7, 247, 248, 249, 250, 251,
                    #        252, 253, 254, 255]
                    # cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.jpg' % epoch), image[0] * 255)
                    # image has three kinds of values: [0, 1, 2]
                    cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.png' % epoch), image[0])
                    if epoch % 1 == 0:
                        print('num %d accuracy: %.6f' % (epoch, acc))
            except tf.errors.OutOfRangeError:
                print('Done testing -- epoch limit reached \n Average accuracy: %.2f%%' % (sum_acc / TEST_SET_SIZE * 100))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
            # coord.request_stop()
            coord.join(threads)
        print('Done testing')

    def predict(self):
        import cv2
        import glob
        import numpy as np
        # ['mv_data_set/test/0.tif', 'mv_data_set/test/1.tif',  ..., 'mv_data_set/test/9.tif']
        predict_file_path = glob.glob(os.path.join(ORIGIN_PREDICT_DIRECTORY, '*.png'))
        print("predict file number:", len(predict_file_path))
        if not os.path.exists(PREDICT_SAVED_DIRECTORY):
            os.mkdir(PREDICT_SAVED_DIRECTORY)
        ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt") # CHECK_POINT_PATH
        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:  # 开始一个会话
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)
            # tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
            all_parameters_saver.restore(sess=sess, save_path=ckpt_path)
            for index, image_path in enumerate(predict_file_path):
                # print "image_path", image_path
                # image = cv2.imread(image_path)
                # print "predict image shape", image.shape
                image = np.reshape(
                    # a=cv2.imread(image_path, flags=0),
                    a=cv2.imread(image_path),
                    newshape=(1, INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL))
                predict_image = sess.run(
                    tf.argmax(input=self.prediction, axis=3),
                    feed_dict={
                        self.input_image: image, self.keep_prob: 1.0, self.lamb: 0.004, self.is_training: False
                    }
                )
                predict_image = np.reshape(predict_image, (OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE))
                # print "predict_image", type(predict_image)
                # print np.unique(predict_image)
                # print image_path.split('/')[2], predict_image.shape
                # cv2.imwrite(str(index)+'.png', predict_image)
                cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, image_path.split('/')[2]), predict_image)  # * 255
                # cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.jpg' % index), predict_image)  # * 255
        print('Done prediction')


def main():
    # OSS_PATH = "oss://u-net-samples/"
    # FLAGS.data_dir = OSS_PATH + 'mv_data_set'
    # FLAGS.model_dir = OSS_PATH + 'mv_data_set/saved_models'
    # FLAGS.tb_dir = OSS_PATH + 'mv_data_set/logs'

    net = Unet()

    net.set_up_unet(TRAIN_BATCH_SIZE)
    net.train()

    # net.set_up_unet(VALIDATION_BATCH_SIZE)
    # net.validate()

    # net.set_up_unet(TEST_BATCH_SIZE)
    # net.test()

    # net.set_up_unet(PREDICT_BATCH_SIZE)data
    # net.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    OSS_PATH = "oss://xixiyuanqu/"
    # 数据地址
    parser.add_argument(
        '--data_dir', type=str, default=OSS_PATH+'mv_data_set/',
        help='Directory for storing input data')

    # 模型保存地址
    parser.add_argument(
        '--model_dir', type=str, default=OSS_PATH+'mv_data_set/saved_models',
        help='output model path')

    # 日志保存地址
    parser.add_argument(
        '--tb_dir', type=str, default=OSS_PATH+'mv_data_set/logs',
        help='TensorBoard log path')

    FLAGS, _ = parser.parse_known_args()

    # write_img_to_tfrecords()
    # read_check_tfrecords()
    main()
    # tf.app.run()
    # calculate_unet_input_and_output()
