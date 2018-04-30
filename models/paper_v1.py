import tensorflow as tf

from .ops import batch_norm
from base.base_model import BaseModel
from tensorflow import layers as L


def _net(input_tensor, is_training=True, config={}):

    if config['activation'] == 'prelu':
        activation = tf.nn.leaky_relu
    else:
        activation = tf.nn.relu

    if config['pooling'] == 'avg':
        pooling = tf.nn.avg_pool
    else:
        pooling = tf.nn.max_pool

    # encoder
    net = input_tensor
    enc_lyr_names = ['Enc_1', 'Enc_2', 'Enc_3', 'Enc_4']
    enc_lyr_depth = [64, 128, 256, 512]
    enc_lyrs = {}
    assert len(enc_lyr_names) == len(enc_lyr_depth)
    for idx in range(len(enc_lyr_names)):
        lyr_name = enc_lyr_names[idx]
        # print('input', lyr_name, net)
        with tf.variable_scope(lyr_name):
            # downsampling path
            if lyr_name != 'Enc_1':
                net_pool = pooling(
                    net, ksize=[1, 2, 2, 1], 
                    strides=[1, 2, 2, 1], padding='SAME', 
                    name='{}_DSamp_Pool'.format(lyr_name))
                net_conv = L.conv2d(
                    net, enc_lyr_depth[idx], 
                    [2, 2], strides=2, padding='SAME',
                    name='{}_DSamp_W'.format(lyr_name))
                net_conv = activation(net_conv, name='{}_DSamp_A'.format(lyr_name))
                net_conv = batch_norm(net_conv, is_training, scope='{}_DSamp_BN'.format(lyr_name))
                net = tf.concat(axis=-1, values=[net_pool, net_conv])
                # net = net_pool
            # print('pooling', lyr_name, net)

            # triangle path
            tri = L.conv2d(net, enc_lyr_depth[idx], [3, 3], strides=1, padding='SAME', name='{}_Tri_W1'.format(lyr_name))
            tri = activation(tri, name='{}_Tri_A1'.format(lyr_name))
            tri = batch_norm(tri, is_training, scope='{}_Tri_BN1'.format(lyr_name))
            tri = L.conv2d(tri, enc_lyr_depth[idx], [1, 1], strides=1, padding='SAME', name='{}_Tri_W2'.format(lyr_name))
            tri = batch_norm(tri, is_training, scope='{}_Tri_BN2'.format(lyr_name))

            net = L.conv2d(net, enc_lyr_depth[idx], [3, 3], strides=1, padding='SAME', name='{}_Main_W1'.format(lyr_name))
            net = activation(net, name='{}_Main_A1'.format(lyr_name))
            net = batch_norm(net, is_training, scope='{}_Main_BN1'.format(lyr_name))
            net = L.conv2d(net, enc_lyr_depth[idx], [3, 3], strides=1, padding='SAME', name='{}_Main_W2'.format(lyr_name))
            net = batch_norm(net, is_training, scope='{}_Main_BN2'.format(lyr_name))
            net = tf.add(net, tri, name='{}_TriMainSum'.format(lyr_name))
            net = activation(net, name='{}_Main_A2'.format(lyr_name))
            enc_lyrs[lyr_name] = net
            # print('output', lyr_name, net, end='\n\n')

    # decoder
    dec_lyr_names = ['Dec_3', 'Dec_2', 'Dec_1']
    dec_lyr_depth = [256, 128, 64]
    assert len(dec_lyr_names) == len(dec_lyr_depth)
    for idx in range(len(dec_lyr_names)):
        lyr_name = dec_lyr_names[idx]
        # print('input', lyr_name, net)
        with tf.variable_scope(lyr_name):
            net = L.conv2d_transpose(
                net, dec_lyr_depth[idx], [2, 2], strides=2, padding='SAME', name='{}_USamp_W'.format(lyr_name))
            net = activation(net, name='{}_USamp_A'.format(lyr_name))
            net = batch_norm(net, is_training, scope='{}_USamp_BN'.format(lyr_name))
            # print('upsampling', lyr_name, net)
            
            tri = L.conv2d(net, dec_lyr_depth[idx], [3, 3], strides=1, padding='SAME', name='{}_Tri_W1'.format(lyr_name))
            tri = activation(tri, name='{}_Tri_A1'.format(lyr_name))
            tri = batch_norm(tri, is_training, scope='{}_Tri_BN1'.format(lyr_name))
            tri = L.conv2d(tri, dec_lyr_depth[idx], [1, 1], strides=1, padding='SAME', name='{}_Tri_W2'.format(lyr_name))
            tri = batch_norm(tri, is_training, scope='{}_Tri_BN2'.format(lyr_name))

            net = tf.concat(axis=-1, values=[enc_lyrs['Enc_{}'.format(lyr_name[-1])], net], name='{}_S1_concat'.format(lyr_name))
            net = L.conv2d(net, dec_lyr_depth[idx], [3, 3], strides=1, padding='SAME', name='{}_Main_W1'.format(lyr_name))
            net = activation(net, name='{}_Main_A1'.format(lyr_name))
            net = batch_norm(net, is_training, scope='{}_Main_BN1'.format(lyr_name))
            net = L.conv2d(net, dec_lyr_depth[idx], [3, 3], strides=1, padding='SAME', name='{}_Main_W2'.format(lyr_name))
            net = batch_norm(net, is_training, scope='{}_Main_BN2'.format(lyr_name))
            net = tf.add(net, tri, name='{}_TriMainSum'.format(lyr_name))
            net = activation(net, name='{}_Main_A2'.format(lyr_name))
            # print('output', lyr_name, net, end='\n\n')

    net = L.conv2d(net, 1, [5, 5], strides=1, padding='SAME', name='Output')
    # print('output layer output ==>', net)
    net = tf.nn.sigmoid(net, name='Output_Sigmoid')
    net = tf.reshape(net, (-1, *config['state_size'][:-1]))

    return net


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        print('Using paper ready model v1 ...')
        self.build_model()
        self.init_saver()

    def predict(self, sess, features):
        feed_dict = {self.x: features, self.is_training: False}
        # prediction = tf.argmax(self.logits, axis=-1)
        pred = sess.run(self.logits, feed_dict=feed_dict)
        return pred

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, ())
        state_size = self.config['state_size']
        input_shape = [None, *state_size]
        self.x = tf.placeholder(tf.float32, shape=input_shape)
        self.original_y = tf.placeholder(tf.float32, (None, *state_size[:-1]))
        self.y = tf.reshape(self.original_y, (-1,  *state_size[:-1]))

        self.logits = _net(self.x, config=self.config)

        with tf.name_scope('loss'):
            self.cross_entropy = tf.reduce_mean(tf.reduce_sum(
                tf.nn.weighted_cross_entropy_with_logits(
                    self.y, self.logits, 8)))
            self.train_step = tf.train.AdamOptimizer(
                self.config.learning_rate).minimize(
                    self.cross_entropy, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)