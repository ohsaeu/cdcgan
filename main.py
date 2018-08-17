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
    
    summary_op = tf.summary.merge([
            tf.summary.image("G", g_img),
            tf.summary.image("X", x_img),
            tf.summary.scalar("loss/dloss", d_loss),
            tf.summary.scalar("loss/gloss", g_loss),
        ])
    
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    
    summary_writer = tf.summary.FileWriter(base_dir,sess.graph)
    saver = tf.train.Saver()

    data_files = glob(os.path.join(conf.data_dir, "*"))
    
    iter_counter = 0
    for epoch in range(conf.n_epoch):

        batch_idxs = len(data_files)//conf.n_batch

        for idx in xrange(0, batch_idxs):
            batch_files = data_files[idx*conf.n_batch:(idx+1)*conf.n_batch]
            batch = [get_image(batch_file, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            
            batch_y = list()
            for ldx in xrange(0, conf.n_batch):
                a, b, c, d = batch_files[ldx].split("_")
                batch_y.append(y_label[b])    
                
            if(n_channel == 1):
                b_size, b_height, b_width = batch_images.shape              
                batch_images = batch_images.reshape(b_size,b_height, b_width,1 )
                
            batch_z = np.random.normal(loc=0.0, scale=1.0, size=(conf.n_batch, z_dim)).astype(np.float32)  
            start_time = time.time()

            fetch_dict = {
                "sum_d_opt": d_optim,
                "sum_g_opt": g_optim,
            }

            if np.mod(iter_counter, conf.n_sample_itr) == 0 or np.mod(iter_counter, conf.n_save_epoch*batch_idxs) == 0:
                fetch_dict.update({
                    "summary": summary_op,
                    "sum_g_img": g_img,
            })

            result = sess.run(fetch_dict,feed_dict={z: batch_z, real_images: batch_images, y:batch_y })
            
            if np.mod(iter_counter, conf.n_sample_itr) == 0:
                summary_writer.add_summary(result["summary"], iter_counter)
                summary_writer.flush()
            
            if np.mod(iter_counter, conf.n_save_epoch*batch_idxs) == 0:
                save_img = result['sum_g_img']
                tl.visualize.save_images(save_img, [8 , 8], './{}/train_{:02d}_{:04d}.png'.format(base_dir, epoch, idx))
                saver.save(sess, os.path.join(base_dir,str(epoch)+"_"+"cdcgan_model.ckpt") ) 
                     
            iter_counter += 1
            
    saver.save(sess, os.path.join(base_dir,"final_cdcgan_model.ckpt") )        

if __name__ == '__main__':
    tf.app.run()
