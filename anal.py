import os, sys, pprint, time, datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model import *
from utils import *
from config import get_config
import logging 
import logging.handlers

pp = pprint.PrettyPrinter()
y_vals = ("0.001", "0.005", "0.01", "0.1", "1")
y_label ={
    "0.001" :[1, 0, 0, 0, 0],
    "0.005" :[0, 1, 0, 0, 0],
    "0.01":[0, 0, 1, 0, 0],
    "0.1":[0, 0, 0, 1, 0],
    "1":[0, 0, 0, 0, 1]
    }

def main(_):
    y_dim=5
    
    conf, _ = get_config()
    
    base_dir = os.path.join(conf.log_dir,conf.curr_time)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    logger = logging.getLogger("log") 
    logger.setLevel(logging.INFO) 
    fileHandler = logging.FileHandler(os.path.join(base_dir,  'log.txt')) 
    logger.addHandler(fileHandler) 
    
    pp.pprint(conf)
    
    if conf.is_gray :
        n_channel=1
    else:
        n_channel=3    

    z_dim = conf.n_z

    z = tf.placeholder(tf.float32, [conf.n_batch, z_dim], name='z_noise')
    y = tf.placeholder(tf.float32, shape=[None, y_dim], name='y_label')
     
    real_images =  tf.placeholder(tf.float32, [conf.n_batch, conf.n_img_out_pix, conf.n_img_out_pix,n_channel], name='real_images')

    g_image_h = conf.n_img_out_pix
    g_image_w = conf.n_img_out_pix

    net_g, g_logits = generator_simplified_api(z,y, g_image_h,g_image_w,conf.n_batch,n_channel ,is_train=True, reuse=False)

    net_d, d_logits = discriminator_simplified_api(net_g.outputs, y, is_train=True, reuse=False)

    net_d2, d2_logits = discriminator_simplified_api(real_images, y, is_train=True, reuse=True)

    net_g2, g2_logits = generator_simplified_api(z,y, g_image_h,g_image_w,conf.n_batch,n_channel , is_train=False, reuse=True)
        
    d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
    d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
  
    d_loss = d_loss_real + d_loss_fake
    g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

    d_optim = tf.train.AdamOptimizer(conf.d_lr, beta1=conf.beta1) \
                      .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(conf.g_lr, beta1=conf.beta1) \
                      .minimize(g_loss, var_list=g_vars)

    g_img=tf.clip_by_value((net_g.outputs+1)*127.5 , 0, 255)
    x_img = tf.clip_by_value((real_images+1)*127.5, 0, 255)
    
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(conf.load_dir, conf.ckpt_nm))
          
      
             
    fetch_dict = {
        "sum_g_img": g_img,
    }
    
    for j in xrange(0, 3):
        batch_z = np.random.normal(loc=0.0, scale=1.0, size=(conf.n_batch, z_dim)).astype(np.float32)
        for i in y_vals:   
            batch_y = list() 
            for k in xrange(conf.n_batch):
                batch_y.append(y_label[i])
            result = sess.run(fetch_dict,feed_dict={z: batch_z, y:batch_y }) 
            tl.visualize.save_images(result["sum_g_img"], [8, 8], './{}/anal_{}_{:d}.png'.format(base_dir, i, j))
    

if __name__ == '__main__':
    tf.app.run()
