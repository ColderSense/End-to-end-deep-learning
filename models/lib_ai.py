#!/usr/bin/python
# encoding: utf-8  
import tensorflow as tf
from scipy.misc import imread,imsave
import numpy as np


#tensorflow high layer function
def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def Conv2D(x, kernel_shape, strides, with_bias = True, name="c", padd = "SAME"):
    with tf.variable_scope(name):
        # Create variable named "w"
        weights = tf.get_variable("w", \
                kernel_shape,initializer = tf.truncated_normal_initializer(stddev=0.01))
        # Create variable named "b"
        bias_shape = kernel_shape[3]
        biases = tf.get_variable("b", \
                bias_shape,initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding=padd)
        if with_bias:
            return (conv + biases)
        else:
            return conv

def Deconv2D(x, kernel_shape, output_shape, strides, name="c", padd = "SAME"):
    with tf.variable_scope(name):
        # Create variable named "w"
        weights = tf.get_variable("w", \
                kernel_shape,initializer = tf.truncated_normal_initializer(stddev=0.01))
        # Create variable named "b"
        bias_shape = kernel_shape[2]
        output_shape = output_shape
        output_shape.append(kernel_shape[2])
        output_shape.insert(0,tf.shape(x)[0])
        biases = tf.get_variable("b", \
                bias_shape,initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1, strides, strides, 1], padding=padd)
        return (conv + biases)

def Dense(x,weights_shape, dropout_facor = 1, activation = "None", name = "fc" ):
    with tf.variable_scope(name):
        weights = tf.get_variable("w", \
                weights_shape,initializer = tf.truncated_normal_initializer(stddev=0.01))
        # Create variable named "b
        bias_shape = weights_shape[1]
        biases = tf.get_variable("b", \
                bias_shape,initializer = tf.constant_initializer(0.0))
        fc = tf.matmul(x ,weights) + biases

        if dropout_facor != 1:
            fc = tf.nn.dropout(fc, dropout_facor)
        if activation == "relu":
            return tf.nn.relu(fc)
        if activation == "elu":
            return tf.nn.elu(fc)  
        if activation == "tanh":
            return tf.nn.tanh(fc) 
        if activation == "None":
            return fc  

def resize_img(x, resize_shape):
        x = tf.image.resize_images(x, resize_shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return x

def batch_norm(x, is_training = True, scope='bn', decay=0.9, eps=1e-5):
    with tf.variable_scope(scope):
        n_out = x.get_shape()[-1]
        beta = tf.get_variable(name='beta', shape=[n_out], \
                initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out],\
                 initializer=tf.random_normal_initializer(1.0, 0.02), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema = tf.train.ExponentialMovingAverage(decay=decay)
     
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed
