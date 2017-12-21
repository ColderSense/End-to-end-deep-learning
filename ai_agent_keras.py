# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 20:30:45 2017

@author: Edward
"""
import socket
import time
import numpy as np
import torch
from main_train_steering import model
from lstm_cnn_keras import get_model
import cv2
import tensorflow as tf

# Config the parameters of socket connection
sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.bind(('localhost',10001))  #socket port number
sock.listen(5)                  #maxmium number of clients
print("start server")


if __name__ =="__main__":
    # Load keras model
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    l_train, saver, _, loader, L  = get_model(sess, training_mode=False)
    loader()
    # Main loop
    input_img = np.zeros((10, 160, 800, 3))
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
        print("image has been recevied")
        img = buffer_.reshape((1,160,800,3), order="F")
        
        # Transform datatype to the input of model
        #img = img.transpose(0, 2, 3, 1)
        img = img/127.5 - 1
        input_img[0:9,:,:,:] = input_img[1:10,:,:,:]
        input_img[9,:,:,:] = img
        img_eval = input_img.reshape((1, 10, 160, 800, 3))

        # Caculate the steering angle
        predicted_steers =  L.predict(img_eval, batch_size=1)[0][0]

        steer = str(predicted_steers)
        
        # Send steering angle to matlab
        connection.send(bytes(steer, encoding = "utf8"))
        print(steer, '\n')
        connection.close()
    sock.close()
    sess.close()
