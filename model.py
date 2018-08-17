import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from utils import get_shape, lkrelu

def generator_simplified_api(z, y, g_image_h, g_image_w,n_batch,n_channel, is_train=True, reuse=False):
    
    h2, h4, h8, h16 = int(g_image_h/2), int(g_image_h/4), int(g_image_h/8), int(g_image_h/16)
    w2, w4, w8, w16 = int(g_image_w/2), int(g_image_w/4), int(g_image_w/8), int(g_image_w/16)
    gf_dim = 64 # Dimension of gen filters in first conv layer. [64]
   
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        
        inputs = tf.concat(axis=1, values=[z, y])
    
        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=g_image_h* g_image_h, W_init=w_init,
                act = tf.nn.relu, name='g/h0/nlin')
        
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*h16*w16, W_init=w_init,
                act = tf.identity, name='g/h0/lin')
        net_h0 = ReshapeLayer(net_h0, shape=[-1, h16, w16, gf_dim*8], name='g/h0/reshape')
        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(h8, w8), strides=(2, 2),
                padding='SAME', batch_size=n_batch, act=None, W_init=w_init, name='g/h1/decon2d')
        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')

        net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(h4, w4), strides=(2, 2),
                padding='SAME', batch_size=n_batch, act=None, W_init=w_init, name='g/h2/decon2d')
        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')

        net_h3 = DeConv2d(net_h2, gf_dim, (5, 5), out_size=(h2, w2), strides=(2, 2),
                padding='SAME', batch_size=n_batch, act=None, W_init=w_init, name='g/h3/decon2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = DeConv2d(net_h3, n_channel, (5, 5), out_size=(g_image_h, g_image_w), strides=(2, 2),
                padding='SAME', batch_size=n_batch, act=None, W_init=w_init, name='g/h4/decon2d')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)
    return net_h4, logits

def discriminator_simplified_api(x,y, is_train=True, reuse=False):
    df_dim = 64 # Dimension of discrim filters in first conv layer. [64]
    d_image_h = x.shape[1].value
    #y =  tf.tile(tf.reshape(y, [-1, 1, 1, get_shape(y)[-1]]))
                 
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        #inputs = tf.concat(axis=1, values=[x, y])
    
        net_in = InputLayer(x, name='d/in')
        #net_h0 = DenseLayer(net_in, n_units=d_image_h* d_image_h, W_init=w_init, act = tf.nn.relu, name='d/h0/nlin')
        
        net_h0 = Conv2d(net_in, df_dim, (5, 5), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d/h0/conv2d')

        conv_1_concat_ys = tf.concat([net_h0.outputs, tf.tile(tf.reshape(y, [-1, 1, 1, get_shape(y)[-1]]),
                                                                [1, tf.shape(net_h0.outputs)[1], tf.shape(net_h0.outputs)[2], 1])], axis=3)
        a_1 = lkrelu(conv_1_concat_ys, slope=0.2)
         
        net_in1 = InputLayer(a_1, name='d/in_y')
                
        net_h1 = Conv2d(net_in1, df_dim*2, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h1/conv2d')
        
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

        net_h2 = Conv2d(net_h1, df_dim*4, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h2/batch_norm')

        net_h3 = Conv2d(net_h2, df_dim*8, (5, 5), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity,
                W_init = w_init, name='d/h4/lin_sigmoid')
        logits = net_h4.outputs
        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)
    #return logits, net_h4
    return net_h4, logits
