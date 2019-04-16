from models.base_model import BaseModel
import tensorflow as tf
from models.utils.cnn_utils import batch_normalization, relu, conv_layer, dropout, average_pool, fc_layer, max_pool, \
    global_average_pool, flatten, concatenation


class DenseNet(BaseModel):
    def __init__(self, sess, conf):
        super(DenseNet, self).__init__(sess, conf)
        assert self.conf.num_levels == len(self.conf.num_BBs), "number of levels doesn't match with number of blocks!"
        self.k = conf.growth_rate
        self.trans_out = 3 * self.k
        self.build_network(self.x)
        self.configure_network()

    def build_network(self, x):
        # Building network...
        with tf.variable_scope('DenseNet'):
            net = conv_layer(x, num_filters=2 * self.k, kernel_size=7, stride=2, layer_name='Conv0')
            # net = max_pool(net, pool_size=3, stride=2, name='MaxPool0')

            for l in range(self.conf.num_levels):
                net = self.dense_block(net, num_BBs=self.conf.num_BBs[l], block_name='DB_' + str(l))
                print('DB_{} shape: {}'.format(str(l + 1), net.get_shape()))
                net = self.transition_layer(net, scope='TB_' + str(l))
                print('TD_{} shape: {}'.format(str(l + 1), net.get_shape()))

            # net = self.dense_block(net, num_BBs=32, block_name='Dense_final')
            # print('DB_{} shape: {}'.format(str(l + 2), net.get_shape()))
            net = batch_normalization(net, training=self.is_training, scope='BN_out')
            net = relu(net)
            net = global_average_pool(net)
            net = flatten(net)
            self.features = net
            self.logits = fc_layer(net, num_units=self.conf.num_cls, add_reg=self.conf.L2_reg, layer_name='Fc1')
            # [?, num_cls]
            self.prob = tf.nn.softmax(self.logits)
            # [?, num_cls]
            self.y_pred = tf.to_int32(tf.argmax(self.prob, 1))
            # [?] (predicted labels)

    def bottleneck_block(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_normalization(x, training=self.is_training, scope='BN1')
            x = relu(x)
            x = conv_layer(x, num_filters=4 * self.k, kernel_size=1, layer_name='CONV1')
            x = dropout(x, rate=self.conf.dropout_rate, training=self.is_training)

            x = batch_normalization(x, training=self.is_training, scope='BN2')
            x = relu(x)
            x = conv_layer(x, num_filters=self.k, kernel_size=3, layer_name='CONV2')
            x = dropout(x, rate=self.conf.dropout_rate, training=self.is_training)
            return x

    def transition_layer(self, x, scope):
        with tf.variable_scope(scope):
            x = batch_normalization(x, training=self.is_training, scope='BN')
            x = relu(x)
            x = conv_layer(x, num_filters=int(x.get_shape().as_list()[-1]*self.conf.theta),
                           kernel_size=1, layer_name='CONV')
            x = dropout(x, rate=self.conf.dropout_rate, training=self.is_training)
            x = average_pool(x, pool_size=2, stride=2, name='AVG_POOL')
            return x

    def dense_block(self, input_x, num_BBs, block_name):
        with tf.variable_scope(block_name):
            layers_concat = list()
            layers_concat.append(input_x)
            x = self.bottleneck_block(input_x, scope='BB_' + str(0))
            layers_concat.append(x)
            for i in range(num_BBs - 1):
                x = concatenation(layers_concat)
                x = self.bottleneck_block(x, scope='BB_' + str(i + 1))
                layers_concat.append(x)
            x = concatenation(layers_concat)
            return x
