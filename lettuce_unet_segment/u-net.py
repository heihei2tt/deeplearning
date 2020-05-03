"""
U-Net 是一种全卷积+上采样的结构，而且下采样和上采样是对应的
"""

import tensorflow as tf
from tfrecord_read_write import read_batch_image, write_image_to_tfrecords


# @staticmethod
# 重新复习一遍batch norm的原理: https://www.cnblogs.com/guoyaohua/p/8724433.html
# 为了满足IID，将隐含层的输入分布拉回到mean=0, var=1的分布，此时，以95%的概率落在[-2,2]区间，以68%的概率落在[-1,1]区间，
# 而sigmoid函数的导数值范围为[0, 0.25]，在输入x=0即sigmoid(x)=0.5时，导数最大为0.25，
def batch_normalization(x, is_training, eps=0.01, decay=0.9, affine=True, name="BatchNorm2D"):
    from tensorflow.python.training.moving_averages import assign_moving_average

    with tf.variable_scope(name):
        # x.shape: [batch_size, height, width, depth]
        params_shape = x.shape[-1]
        # tf.get_variable与tf.Variable的区别？
        moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                     trainable=False)

        def mean_var_with_update():
            mean_this_batch, variance_this_batch = tf.nn.moments(x, list[x.shape], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean_this_batch, decay),
                                          assign_moving_average((moving_var, variance_this_batch, decay))]):
                return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

        # 如果is_training is True, 使用不断更新的mean和var， 否则使用全局的mean和var
        mean, variance = tf.cond(is_training, mean_var_with_update(), lambda: (moving_mean, moving_var))

        if affine:
            # 使用beta和gamma缩放, x = (x - mean) / (tf.sqrt(var) + eps), y = gamma * x + beta
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape, params_shape, initializer=tf.ones_initializer)
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                               variance_epsilon=eps)
        else:
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                               variance_epsilon=eps)

        return normed


class UNet:
    def __init__(self):
        self.input_image = None
        self.input_label = None
        self.keep_prob = None
        self.is_training = None
        self.lamb = None
        self.w = {}
        self.b = {}
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contract_layer = {}
        self.prediction = None
        self.loss = 0
        self.loss_mean = 0
        self.loss_all = 0
        self.correct_prediction = None
        self.accuracy = 0
        self.train_step = None

    def init_weights(self, shape, name):
        """
        :param shape: 输入图像或feature map的height * width * depth
        :param name:
        :return:
        """
        with tf.name_scope("init_weights"):
            stddev = tf.sqrt(x=2.0 / (shape[0] * shape[1] * shape[2]))
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape,
                                                              stddev=stddev,
                                                              dtype=tf.float32),
                            name=name)
            # 将权重的l2范数作为loss中的正则化项
            tf.add_to_collection(name='loss', value=tf.contrib.layers.l2_regularizer(self.lamb)(w))
            return w

    def init_bias(self, shape, name):
        with tf.name_scope("init_bias"):
            return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

    # UNet的左边部分：卷积+下采样
    def build_conv_layer(self, input_feat_map, keep_prob, kernal1, kernal2, layer_idx, is_training):
        """
        卷积+bn+relu + 卷积+bn+relu + max_pool
        :param input_feat_map:
        :param keep_prob:
        :param kernal1: 第1个卷积层conv1的卷积核的shape
        :param kernal2: 第2个卷积层conv2的卷积核的shape
        :param layer_idx: 层号
        :param is_training:
        :return:
        """
        """建立UNet的卷积（下采样）层"""
        with tf.name_scope('layer_' + str(layer_idx)):
            # conv1
            self.w['w' + str(2 * layer_idx - 1)] = self.init_weights(shape=kernal1, name='w_' + str(2 * layer_idx - 1))
            self.b['b' + str(2 * layer_idx - 1)] = self.init_bias(shape=kernal1[3], name='b_' + str(2 * layer_idx - 1))
            # 弄清楚卷积的计算过程
            result_conv_1 = tf.nn.conv2d(input=input_feat_map, filter=self.w['w' + str(2 * layer_idx - 1)],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME',
                                         name='conv_1')
            result_conv_b_1 = tf.nn.bias_add(result_conv_1, self.b['b' + str(2 * layer_idx - 1)])
            normed_batch_1 = batch_normalization(x=result_conv_b_1, is_training=is_training,
                                                 name='layer' + str(layer_idx) + '_conv_1')
            result_relu_1 = tf.nn.relu(normed_batch_1, name='relu_1')
            # conv2
            self.w['w' + str(2 * layer_idx)] = self.init_weights(shape=kernal2, name='w_' + str(2 * layer_idx))
            self.b['b' + str(2 * layer_idx)] = self.init_bias(shape=kernal2[3], name='b_' + str(2 * layer_idx))
            result_conv_2 = tf.nn.conv2d(input=result_relu_1, filter=self.w['w' + str(2 * layer_idx)],
                                         strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            result_conv_b_2 = tf.nn.bias_add(result_conv_2, self.b['b' + str(2 * layer_idx)])
            normed_batch_2 = batch_normalization(x=result_conv_b_2, is_training=is_training,
                                                 name='layer_' + str(layer_idx) + '_conv_2')
            result_relu_2 = tf.nn.relu(features=normed_batch_2, name='relu_2')

            if layer_idx == 5:
                # bottom层，不做max_pool, 也不用保存contract_layer
                return result_relu_2

            # 该层结果临时保存，供上采样时使用
            self.result_from_contract_layer['layer_' + str(layer_idx)] = result_relu_2
            # maxpool
            result_maxpool = tf.nn.max_pool(value=result_relu_2, ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1], padding='VALID', name='maxpool')
            # dropout
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=keep_prob)

            return result_dropout

    def copy_crop_merge(self, contract_layer_feat_map, upsampling_layer_feat_map):
        """
        这部分是UNet网络的核心，将下采样与上采样拼接
        :param contract_layer_feat_map: 卷积+下采样层的feature map
        :param upsampling_layer_feat_map: 卷积+上采样层feature map
        :return:
        """
        contract_shape = tf.shape(contract_layer_feat_map)
        upsample_shape = tf.shape(upsampling_layer_feat_map)
        contract_part = tf.slice(input=contract_layer_feat_map,
                                 begin=[0,
                                        (contract_shape[1] - upsample_shape[1]) // 2,
                                        (contract_shape[2] - upsample_shape[2]) // 2,
                                        0],
                                 size=[upsample_shape[0],
                                       upsample_shape[1],
                                       upsample_shape[2],
                                       upsample_shape[3]]
                                 )
        return tf.concat(values=[contract_part, contract_layer_feat_map])

    def build_upsample_layer(self, input_feat_map, kernal1, kernal2, kernal3, out_shape, layer_idx, is_training,
                             keep_prob):
        """
        首先，将input_feat_map与对应的下采样卷积的feature map剪裁、拼接;
        然后,做两个下采样卷积;
        最后,反卷积.
        :param input_feat_map:
        :param kernal1: 卷积核1
        :param kernal2: 卷积核2
        :param kernal3: 上采样卷积核
        :param layer_idx: 层号
        :return:
        """
        with tf.name_scope('layer_' + str(layer_idx)):
            # 因为UNet是对称结构的，7～3， 6～4，所以用10-layer_idx
            merged_fmap = self.copy_crop_merge(self.result_from_contract_layer['layer_' + str(10 - layer_idx)],
                                               input_feat_map)

            # conv1
            self.w['w' + str(2 * layer_idx)] = self.init_weights(shape=kernal1, name='w' + str(2 * layer_idx))
            self.b['b' + str(2 * layer_idx)] = self.init_bias(shape=kernal1[3], name='b' + str(2 * layer_idx))
            res_conv1 = tf.nn.conv2d(input=merged_fmap, filter=self.w['w' + str(2 * layer_idx)], strides=[1, 1, 1, 1],
                                     padding='SAME', name='conv1')
            res_conv_b1 = tf.nn.bias_add(res_conv1, self.b['b' + str(2 * layer_idx)])
            res_normed_conv1 = batch_normalization(res_conv_b1, is_training=is_training,
                                                   name='layer_' + str(layer_idx) + '_conv1')
            res_relu1 = tf.nn.relu(features=res_normed_conv1, name='relu1')

            # conv2
            self.w['w' + str(2 * layer_idx + 1)] = self.init_weights(shape=kernal2, name='w' + str(2 * layer_idx + 1))
            self.b['b' + str(2 * layer_idx + 1)] = self.init_bias(shape=kernal2[3], name='b' + str(2 * layer_idx + 1))
            res_conv2 = tf.nn.conv2d(input=res_relu1, filter=self.w['w' + str(2 * layer_idx + 1)], strides=[1, 1, 1, 1],
                                     padding='SAME', name='conv2')
            res_conv_b2 = tf.nn.bias_add(res_conv2, self.b['b' + str(2 * layer_idx + 1)])
            res_normed_conv2 = batch_normalization(x=res_conv_b2, is_training=is_training,
                                                   name='layer_' + str(layer_idx) + '_conv2')
            res_relu2 = tf.nn.relu(res_normed_conv2)

            # 如果是第9层，则不做上采样了
            if layer_idx == 9:
                return res_relu2

            # upsample
            self.w['w' + str(2 * layer_idx + 2)] = self.init_weights(shape=kernal3, name='w' + str(2 * layer_idx + 2))
            self.b['b' + str(2 * layer_idx + 2)] = self.init_bias(shape=kernal3[3], name='b' + str(2 * layer_idx + 2))
            res_upsample = tf.nn.conv2d_transpose(value=res_relu2, filter=self.w['w' + str(2 * layer_idx + 2)],
                                                  output_shape=out_shape, strides=[1, 2, 2, 1],
                                                  padding='VALID', name='upsample' + str(layer_idx))
            res_up_b = tf.nn.bias_add(res_upsample, self.b['b' + str(2 * layer_idx + 2)])
            res_normed_batch = batch_normalization(res_up_b, is_training)
            res_relu3 = tf.nn.relu(res_normed_batch)

            # dropout
            return tf.nn.dropout(res_relu3, keep_prob=keep_prob)

    def setup_unet(self, batch_size, class_num):
        # https://blog.csdn.net/Formlsl/article/details/80373200
        """用的是VGG16网络"""
        with tf.name_scope('input'):
            # 假设input_image的shape=[batch_size, 128, 128, 3]
            self.input_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 128, 128, 3], name='input_images')
            self.input_label = tf.placeholder(dtype=tf.int32, shape=[batch_size, 64, 64], name='input')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.lamb = tf.placeholder(dtype=tf.float32, name='lamb')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            # normed_batch = batch_normalization(x=self.input_image, is_training=self.is_training, name='input')

            # layer1
            conv_relu_layer1 = self.build_conv_layer(self.input_image, self.keep_prob, kernal1=[3, 3, 3, 64],
                                                     kernal2=[3, 3, 64, 64], layer_idx=1, is_training=self.is_training)
            # layer2
            conv_relu_layer2 = self.build_conv_layer(conv_relu_layer1, self.keep_prob, kernal1=[3, 3, 64, 128],
                                                     kernal2=[3, 3, 128, 128], layer_idx=2,
                                                     is_training=self.is_training)
            # layer3
            conv_relu_layer3 = self.build_conv_layer(conv_relu_layer2, self.keep_prob, kernal1=[3, 3, 128, 256],
                                                     kernal2=[3, 3, 256, 256], layer_idx=3,
                                                     is_training=self.is_training)
            # layer4
            conv_relu_layer4 = self.build_conv_layer(conv_relu_layer3, self.keep_prob, kernal1=[3, 3, 256, 512],
                                                     kernal2=[3, 3, 512, 512], layer_idx=4,
                                                     is_training=self.is_training)
            # layer5: bottom layer
            conv_relu_layer5 = self.build_conv_layer(conv_relu_layer4, self.keep_prob, kernal1=[3, 3, 512, 1024],
                                                     kernal2=[3, 3, 1024, 1024], layer_idx=5,
                                                     is_training=self.is_training)
            # upsample
            """
            conv2d_transpose(input, filter, output_shape, strides, padding='VALID', name=None)
            filter: [filter_height, filter_width, <big> out_channel, in_channel </big>]
            output_shape: a 1-D Tensor or a N-D shape list, e.g., [batch, out_height, out_width, out_channel]
            """
            self.w['w11'] = self.init_weights(shape=[2, 2, 512, 1024], name='w11')
            self.b['b11'] = self.init_bias(shape=[512], name='b11')
            result_up = tf.nn.conv2d_transpose(value=conv_relu_layer5,
                                               filter=self.w['w11'],
                                               output_shape=[batch_size, 14, 14, 512],
                                               strides=[1, 2, 2, 1],
                                               padding='VALID',
                                               name='UpSample')
            result_up_b = tf.nn.bias_add(result_up, self.b['b11'])
            normed_batch = batch_normalization(x=result_up_b, is_training=self.is_training, name='layer5_conv_up')
            activated_normed_batch = tf.nn.relu(features=normed_batch, name='activated_normed_batch')
            dropped_normed_batch = tf.nn.dropout(x=activated_normed_batch, keep_prob=self.keep_prob)

            # layer6
            layer6_upsample = self.build_upsample_layer(input_feat_map=dropped_normed_batch, kernal1=[3, 3, 1024, 512],
                                                        kernal2=[3, 3, 512, 512],
                                                        kernal3=[2, 2, 256, 512], out_shape=[batch_size, 28, 28, 256],
                                                        layer_idx=6, is_training=self.is_training,
                                                        keep_prob=self.keep_prob)
            # layer7
            layer7_upsample = self.build_upsample_layer(input_feat_map=layer6_upsample, kernal1=[3, 3, 512, 256],
                                                        kernal2=[3, 3, 256, 256],
                                                        kernal3=[2, 2, 128, 256], out_shape=[batch_size, 56, 56, 128],
                                                        layer_idx=7, is_training=self.is_training,
                                                        keep_prob=self.keep_prob)
            # layer8
            layer8_upsample = self.build_upsample_layer(input_feat_map=layer7_upsample, kernal1=[3, 3, 256, 128],
                                                        kernal2=[3, 3, 128, 128],
                                                        kernal3=[2, 2, 64, 128], out_shape=[batch_size, 112, 112, 64],
                                                        layer_idx=8, is_training=self.is_training,
                                                        keep_prob=self.keep_prob)

            # layer9, 实际上layer9不做上采样
            layer9_upsample = self.build_upsample_layer(input_feat_map=layer8_upsample, kernal1=[3, 3, 128, 64],
                                                        kernal2=[3, 3, 64, 64],
                                                        kernal3=[], out_shape=[],
                                                        layer_idx=9, is_training=self.is_training,
                                                        keep_prob=self.keep_prob)
            # class_num是像素类别数
            self.w['w23'] = self.init_weights(shape=[1, 1, 64, class_num])
            self.b['b23'] = self.init_bias(shape=[class_num])
            layer9_conv3 = tf.nn.conv2d(input=layer9_upsample, filter=[1, 1, 64, class_num], strides=[1, 1, 1, 1], padding='VALID')
            layer9_conv3_b = tf.nn.bias_add(layer9_conv3, self.b['b23'])
            layer9_normed_batch = batch_normalization(x=layer9_conv3_b, is_training=self.is_training, name='layer9_norm')

            self.prediction = layer9_normed_batch

            return self.prediction

    def train(self, train_set_path, ckpt_path, batch_size):
        with tf.name_scope('softmax_loss'):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='loss', value=self.loss_mean)
            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        with tf.name_scope('accuracy'):
            # axis=3 就是depth？
            self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
            # 强制转换成浮点数
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        with tf.name_scope('gradient_descent'):
            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

        train_image_file_queue = tf.train.string_input_producer(string_tensor=tf.train.match_filenames_once(train_set_path),
                                                                num_epochs=10,
                                                                shuffle=True)
        batch_images, batch_labels = read_batch_image(train_image_file_queue, batch_size)

        # summary是什么意思？
        tf.summary.scalar('loss', self.loss_mean)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary = tf.summary.merge_all()
        all_parameters_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # 日志保存地址为./
            summary_writer = tf.summary.FileWriter('./', sess.graph)
            tf.summary.FileWriter('./model', sess.graph)

            # 线程管理器
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                epoch = 1
                while not coord.should_stop():
                    sample, label = sess.run([batch_images, batch_labels])
                    loss_, accuracy_, summary_str = sess.run(
                        [self.loss_mean, self.accuracy, merged_summary],
                        feed_dict={
                            self.input_image: sample, self.input_label: label, self.keep_prob: 0.7, self.lamb: 0.001,
                            self.is_training: True
                        }
                    )
                    summary_writer.add_summary(summary_str, epoch)
                    print('epoch %d, loss: %.6f and accuracy: %.6f' % (epoch, loss_, accuracy_))
                    sess.run(self.train_step,
                             feed_dict={
                                self.input_image: sample, self.input_label: label, self.keep_prob: 0.7, self.lamb: 0.001,
                                self.is_training: True}
                             )
                    epoch += 1
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached.')
            finally:
                all_parameters_saver.save(sess, save_path=ckpt_path)
                coord.request_stop()
            coord.join(threads)
        print('Done training.')


































































