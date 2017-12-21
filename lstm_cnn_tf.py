#!/usr/bin/env python
# encoding: utf-8  
import tensorflow as tf
from scipy.misc import imread,imsave
import numpy as np
import sys
import cv2
from data_gen.server import client_generator
import os
import models.lib_ai as lib
from datetime import datetime
import argparse
from tensorflow.contrib import rnn


def gen(hwm, host, port):
    for tup in client_generator(hwm=hwm, host=host, port=port):
        X, Y, _ = tup
        Y = Y[:, -1]
        #print Z
        if X.shape[1] == 1:  # no temporal context
          X = X[:, -1]
        #X = X.transpose(0, 2, 3, 1)
        #print("image shape", X.shape)
        X = X/127.5 - 1
        yield X, Y

def model(inputs, keep_prob, batch_size=32,reuse=False):
    with tf.variable_scope("e2e"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        # Paramters
        time_len = 5
        z_dim = 512
        hidden_size = 512
        layer_num = 2
        
        # Convlution layers
        x_in = tf.reshape(inputs, [time_len*batch_size, 160,800,3])
        x1 = lib.Conv2D(x_in, [8, 8, 3, 16], 5,  name='C1')
        x1 = tf.nn.relu(x1, name="relu_c1")

        x2 = lib.Conv2D(x1, [5, 5, 16, 32], 2,  name='C2')
        x2 = tf.nn.relu(x2, name="relu_c2")

        x3 = lib.Conv2D(x2, [5, 5, 32, 48], 2, name='C3')
        x3 = tf.nn.relu(x3, name="relu_c3")

        x4 = lib.Conv2D(x3, [5, 5, 48, 64], 2, name='C4')
        x4 = tf.nn.relu(x4, name="relu_c4")       

        x5 = lib.Conv2D(x4, [3, 3, 64, 128], 2, name='C5')   
        x5 = tf.nn.relu(x5, name="relu_c5")  

        x6 = lib.Conv2D(x5, [3, 3, 128, 128], 2, name='C6')    
        x6 = tf.reshape(x6, [-1, 1*5*128])
        x6 = tf.nn.dropout(x6, keep_prob)
        x6 = tf.nn.relu(x6, name="relu_6")
        x6_ = tf.reshape(x6, [-1, 1, 5, 128])

        x_mid = lib.Dense(x6, [640, z_dim], dropout_facor=keep_prob, activation="tanh", name="fc_mid")
        # Filter map
        dim_0 = tf.shape(x6_)[0]
        filter_aver1 = tf.reduce_mean(x1, 3)
        print(tf.shape(filter_aver1))
        filter_aver1 = tf.reshape(filter_aver1, (dim_0, 32, 160, 1))
        filter_aver2 = tf.reduce_mean(x2, 3)
        filter_aver2 = tf.reshape(filter_aver2, (dim_0, 16, 80, 1))
        filter_aver3 = tf.reduce_mean(x3, 3)
        filter_aver3 = tf.reshape(filter_aver3, (dim_0, 8, 40, 1))
        filter_aver4 = tf.reduce_mean(x4, 3)
        filter_aver4 = tf.reshape(filter_aver4, (dim_0, 4, 20, 1))
        filter_aver5 = tf.reduce_mean(x5, 3)
        filter_aver5 = tf.reshape(filter_aver5, (dim_0, 2, 10, 1))
        filter_aver6 = tf.reduce_mean(x6_, 3)
        filter_aver6 = tf.reshape(filter_aver6, (dim_0, 1, 5, 1))

        # Backprop
        w1 = tf.get_variable('w1', initializer=tf.ones(([3, 3, 1, 1])))
        deconv1 = tf.nn.conv2d_transpose(filter_aver6,w1,[dim_0,2,10,1], strides=[1,2,2,1])
        #deconv1 = lib.resize_img(filter_aver5, [7, 12])
        #deconv1 = tf.nn.conv2d(deconv1, w1, [1, 1, 1, 1], padding="VALID")
        back_conv1 = tf.multiply(deconv1, filter_aver5)

        w2 = tf.get_variable('w2', initializer=tf.ones(([3, 3, 1, 1])))
        deconv2 = tf.nn.conv2d_transpose(back_conv1,w2,[dim_0,4,20,1],strides=[1,2,2,1])
        #deconv2 = lib.resize_img(back_conv1, [20, 40])
        #deconv2 = tf.nn.conv2d(deconv2, w2, [1, 2, 2, 1], padding="SAME")
        back_conv2 = tf.multiply(deconv2, filter_aver4)

        w3 = tf.get_variable('w3', initializer=tf.ones(([5, 5, 1, 1])))
        deconv3 = tf.nn.conv2d_transpose(back_conv2,w3,[dim_0,8,40,1], strides=[1,2,2,1])
        #deconv3 = lib.resize_img(back_conv2, [40, 80])
        #deconv3 = tf.nn.conv2d(deconv3, w3, [1, 2, 2, 1], padding="SAME")
        back_conv3 = tf.multiply(deconv3, filter_aver3)

        w4 = tf.get_variable('w4', initializer=tf.ones(([5, 5, 1, 1])))
        deconv4 = tf.nn.conv2d_transpose(back_conv3,w4,[dim_0,16,80,1], strides=[1,2,2,1])
        #deconv4 = lib.resize_img(back_conv3, [80, 160])
        #deconv4 = tf.nn.conv2d(deconv4, w4, [1, 2, 2, 1], padding="SAME")
        back_conv4 = tf.multiply(deconv4, filter_aver2)

        w5 = tf.get_variable('w5', initializer=tf.ones(([5, 5, 1, 1])))
        deconv5 = tf.nn.conv2d_transpose(back_conv4,w5,[dim_0,32,160,1], strides=[1,2,2,1])
        #deconv5 = lib.resize_img(back_conv4, [640, 1280])
        #deconv5 = tf.nn.conv2d(deconv5, w5, [1, 4, 4, 1], padding="SAME")
        back_conv5 = tf.multiply(deconv5, filter_aver1)

        w6 = tf.get_variable('w6', initializer=tf.ones(([8, 8, 1, 1])))
        deconv6 = tf.nn.conv2d_transpose(back_conv5,w6,[dim_0,160,800,1], strides=[1,5,5,1])


        # Lstms
        print(x_mid.get_shape())
        X = tf.reshape(x_mid, [-1, time_len, z_dim])

        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
        init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
        h_state = outputs[:, -1, :]

        x_out = lib.Dense(h_state, [512,1], name="fc_out")
    return x_out, deconv6 

def get_model(sess, training_mode=False, batch_size=32, time_len=5):
    # Directory
    save_dir = './results/model_weights/commaai'
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logdir = './logs/commaai/'+time+''

    # Placeholder
    keep_prob = tf.placeholder(tf.float32, name='kp1')
    I = tf.placeholder(tf.float32, [batch_size, time_len, 160, 800, 3], name='image')
    I_img = tf.placeholder(tf.float32, [1, time_len, 160, 800, 3], name='image_eval')
    S = tf.placeholder(tf.float32, shape=[batch_size, 1], name='steer')

    # Model
    S_, _ = model(I, keep_prob, batch_size=batch_size)
    _, vis_back = model(I_img, keep_prob, batch_size=1, reuse=True)
    img_backcnn = tf.multiply(tf.add(vis_back, 0), 255)

    # Loss
    loss = tf.reduce_mean(tf.square(S_ - S))
    var_list = [v for v in tf.trainable_variables() \
                if (v.name.startswith('e2e'))]   
    optim = tf.train.AdamOptimizer(learning_rate=5e-4, \
                name="Adam_e2e").minimize(loss, var_list=var_list)

    # Summary
    sum_loss = tf.summary.scalar("loss", loss)
    sum_loss_test = tf.summary.scalar("loss_test", loss)
    sum_img = tf.summary.image("Back_prop", img_backcnn, max_outputs=6)

    # Initialize
    sess.run(tf.global_variables_initializer())
    # mode
    saver = tf.train.Saver()
    if training_mode:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = tf.summary.FileWriter(logdir, sess.graph)

    # Test images
    img_test = []
    for x_ in range(5):
        img_test.append(imread("./samples/"+"sample_"+str(x_)+".jpg")/127.5 - 1)
    img_test = np.array(img_test)
    img_test = img_test.reshape(1,5,160,800,3)

    def batch_gen(it):
        batch = next(it) #get data for training from data server
        batch = list(batch)
        batch_x = batch[0]
        batch_y = batch[1]
        return batch_x, batch_y

    def f_train(images, labels, counter, sess=sess):
        _, loss = sess.run([optim, sum_loss], feed_dict={I: images, S: labels,  keep_prob: 0.5})
        writer.add_summary(loss, counter)

    def f_test(images, labels, counter, sess=sess):
        loss = sum_loss_test.eval(feed_dict={I: images, S: labels, keep_prob: 1})
                    
        writer.add_summary(loss, counter)
        sum_img_tf = sum_img.eval(feed_dict={I_img: img_test, keep_prob: 1})
        writer.add_summary(sum_img_tf, counter)

        save_img = img_backcnn.eval(feed_dict={I_img: img_test, keep_prob: 1})[1,:,:,:]
        save_img[save_img<0] = 0
        save_img[save_img>255] = 255
        save_img = save_img.astype(np.uint8)
        print(save_img)
        cv2.imwrite('test.png', save_img)

    def f_save(sess=sess):
        saver.save(sess, save_dir+'/model.ckpt')
        print("tensorflow model has been saved")

    def f_load(sess=sess):
        ckpt = tf.train.get_checkpoint_state(save_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('tensorflow model has been loaded from:',save_dir)

    def f_predict(images, sess=sess):

        out, color_img = sess.run([S_, img_backcnn], feed_dict={I: images, I_img: images, keep_prob: 1})
        return out[0][0], color_img
    return batch_gen, f_train, f_test, f_save, f_load, f_predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=100, help='Batch size.')
    parser.add_argument('--model', type=str, default='autoencoder', help='model to train')
    parser.add_argument('--iters', type=int, default='50000', help='model to train')
    args = parser.parse_args()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Get model
        gen_batch, train_f, test_f, save_f, load_f, predict_f = get_model(sess, training_mode=True)

        # Generator object
        it = gen(20, "localhost", port=5557)
        it_val = gen(20, "localhost", port=5556)

        # Load weights
        load_f()

        # Predict
        '''
        img = imread("./samples/"+"sample_0"+".jpg")/127.5 - 1
        img = img.reshape(1,160,800,3)
        angle = predict_f(img)
        print(angle)
        '''

        # Training loop
        for epoch in range(args.iters):
            batch_x, batch_y = gen_batch(it)
            train_f(batch_x, batch_y, epoch)
            epoch = epoch+1

            if epoch % 100 == 0:
                batch_x, batch_y = gen_batch(it_val)
                test_f(batch_x, batch_y, epoch)
                save_f()
                print("epoch: %d" % epoch)