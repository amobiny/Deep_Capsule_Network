from models.base_model import BaseModel
from models.capsule_layers.Conv_Caps import ConvCapsuleLayer
from models.capsule_layers.FC_Caps import FCCapsuleLayer
from keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.image_ops_impl import ResizeMethod


class CapsNet(BaseModel):
    def __init__(self, sess, conf):
        super(CapsNet, self).__init__(sess, conf)
        self.build_network(self.x)
        if self.conf.mode != 'train_sequence':
            self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('CapsNet'):
            # Layer 1: A 2D conv layer
            conv1 = layers.Conv2D(filters=128, kernel_size=5, strides=2,
                                  padding='valid', activation='relu', name='conv1')(x)

            # Reshape layer to be 1 capsule x caps_dim(=filters)
            _, H, W, C = conv1.get_shape()
            conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

            # Layer 2: Convolutional Capsule
            primary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=8, caps_dim=16, strides=2, padding='same',
                                            routings=3, name='primarycaps')(conv1_reshaped)

            # Layer 3: Convolutional Capsule
            secondary_caps = ConvCapsuleLayer(kernel_size=5, num_caps=8, caps_dim=16, strides=2, padding='same',
                                              trainable=self.conf.trainable, routings=3, name='secondarycaps')(primary_caps)
            _, H, W, D, dim = secondary_caps.get_shape()
            sec_cap_reshaped = layers.Reshape((H.value * W.value * D.value, dim.value))(secondary_caps)

            # Layer 4: Fully-connected Capsule
            self.digit_caps = FCCapsuleLayer(num_caps=self.conf.num_cls, caps_dim=self.conf.digit_caps_dim,
                                             routings=3, name='digitcaps')(sec_cap_reshaped)
            # [?, 10, 16]

            epsilon = 1e-9
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.digit_caps), axis=2, keep_dims=True) + epsilon)
            # [?, 10, 1]
            self.act = tf.reshape(self.v_length, (self.conf.batch_size, self.conf.num_cls))
            self.prob = tf.nn.softmax(self.act)

            y_prob_argmax = tf.to_int32(tf.argmax(self.v_length, axis=1))
            # [?, 1]
            self.y_pred = tf.squeeze(y_prob_argmax)
            # [?] (predicted labels)

            if self.conf.add_recon_loss:
                self.mask()
                self.decoder()

            if self.conf.before_mask:
                self.features = self.digit_caps
            else:
                self.features = self.output_masked

    def decoder(self):
        with tf.variable_scope('Deconv_Decoder'):
            cube_size = np.sqrt(self.conf.digit_caps_dim).astype(int)
            decoder_input = tf.reshape(self.output_masked, [-1, self.conf.num_cls, cube_size, cube_size])
            cube = tf.transpose(decoder_input, [0, 2, 3, 1])
            # res1 = Deconv2D(cube,
            #                 filter_size=2,
            #                 num_filters=8,
            #                 stride=1,
            #                 layer_name="deconv_0",
            #                 out_shape=[self.conf.batch_size, 7, 7, 8])
            # res1 = Deconv2D(res1,
            #                 filter_size=2,
            #                 num_filters=16,
            #                 stride=2,
            #                 layer_name="deconv_1",
            #                 out_shape=[self.conf.batch_size, 13, 13, 16])
            # res2 = Deconv2D(res1,
            #                 filter_size=4,
            #                 num_filters=16,
            #                 stride=2,
            #                 layer_name="deconv_2",
            #                 out_shape=[self.conf.batch_size, 26, 26, 16])
            # self.decoder_output = Deconv2D(res2,
            #                                filter_size=4,
            #                                num_filters=1,
            #                                stride=2,
            #                                layer_name="deconv_3",
            #                                out_shape=[self.conf.batch_size, 51, 51, 1])

            conv_rec1_params = {"filters": 8,
                                "kernel_size": 2,
                                "strides": 1,
                                "padding": "same",
                                "activation": tf.nn.relu}

            conv_rec2_params = {"filters": 16,
                                "kernel_size": 3,
                                "strides": 1,
                                "padding": "same",
                                "activation": tf.nn.relu}

            conv_rec3_params = {"filters": 16,
                                "kernel_size": 3,
                                "strides": 1,
                                "padding": "same",
                                "activation": tf.nn.relu}

            conv_rec4_params = {"filters": 1,
                                "kernel_size": 3,
                                "strides": 1,
                                "padding": "same",
                                "activation": None}

            conv_rec1 = tf.layers.conv2d(cube, name="conv1_rec", **conv_rec1_params)
            res1 = tf.image.resize_images(conv_rec1, (8, 8), method=ResizeMethod.NEAREST_NEIGHBOR)
            conv_rec2 = tf.layers.conv2d(res1, name="conv2_rec", **conv_rec2_params)
            res2 = tf.image.resize_images(conv_rec2, (17, 17), method=ResizeMethod.NEAREST_NEIGHBOR)
            conv_rec3 = tf.layers.conv2d(res2, name="conv3_rec", **conv_rec3_params)
            res3 = tf.image.resize_images(conv_rec3, (51, 51), method=ResizeMethod.NEAREST_NEIGHBOR)
            self.decoder_output = tf.layers.conv2d(res3, name="conv4_rec", **conv_rec4_params)

