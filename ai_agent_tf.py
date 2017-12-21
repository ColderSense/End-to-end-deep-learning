# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:30:45 2017

@author: Edward
"""
import socket
import time
import cv2
import numpy as np
import pygame
from scipy.misc import imread,imsave
from comaai_steering import get_model
import tensorflow as tf


sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.bind(('localhost',10001))  #socket port number
sock.listen(5)                  #maxmium number of clients

print("start server")

def normalize_std(img):
  '''Normalize image by making its standard deviation = 1.0'''
  with tf.name_scope('normalize'):
    std = tf.sqrt(tf.reduce_mean(tf.square(img), axis = (1, 2), keep_dims=True))
    return img/tf.maximum(std, 1e-7)

# Config of tensorflow
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  _, _, _, _, load_f, predict_f = get_model(sess, batch_size=1)

  # Load weights
  load_f()

  input_img = np.zeros((5, 160, 800, 3))
  while True:
    connection, address = sock.accept()
    #print("client ip is :", address)
    # Initialize parameters of buffer
    buf_size = 1024
    filesize = 384000   #maxmium temp bytes
    recvd_size = 0      #current received bytes
    buffer_ = ()         #buffer_ for store the image bytes
    data = ()            #temp variable
    while True:
      if (filesize - recvd_size) >= buf_size :
          data = connection.recv(buf_size)
          data = np.frombuffer(data, dtype='uint8')
          buffer_ = np.concatenate((buffer_,data))
          recvd_size += len(data)
          #print (filesize - recvd_size)
      elif (filesize - recvd_size) >=0 and (filesize - recvd_size) < buf_size:
          data = connection.recv(filesize - recvd_size)
          data = np.frombuffer(data, dtype='uint8')
          buffer_ = np.concatenate((buffer_,data))
          #if filesize == recvd_size:
          #print(recvd_size)
          break
    print("image has been recevied\n")

    # Get image data
    img = buffer_.reshape((160,800,3), order="F")
    img = img/127.5 - 1
    input_img[0:4,:,:,:] = input_img[1:5,:,:,:]
    input_img[4,:,:,:] = img
    img_eval = input_img.reshape((1, 5, 160, 800, 3))

    # Caculate the steering angle
    predicted_steers, color_img = predict_f(img_eval)

    # Show point area gray map
    color_img = color_img.reshape(5,160,800,1)
    area_img = color_img[4,:,:,:]
    area_img = area_img.astype(np.uint8)
    cv2.imshow('point_area_view', area_img)
    cv2.waitKey(1)
    
    # Send steering angle to matlab
    steer = str(predicted_steers)
    connection.send(bytes(steer, encoding = "utf8"))
    print(predicted_steers)
    connection.close()
  sock.close()
