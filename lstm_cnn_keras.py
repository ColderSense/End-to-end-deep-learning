from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Activation, LSTM, \
      Conv2D, LeakyReLU, Flatten, Dropout, BatchNormalization as BN
import tensorflow as tf
from keras import backend as K
import os
from data_gen.server import client_generator
from keras import initializers
from functools import partial
import numpy as np 
from datetime import datetime
from keras.layers.wrappers import TimeDistributed


normal = partial(initializers.normal, scale=.02)


def mean_normal(shape, mean=1., scale=0.02, name=None):
    return K.variable(np.random.normal(loc=mean, scale=scale, size=shape), name=name)



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

def cnn_lstm(batch_size, df_dim, z_dim, time_len=5, img_shape=(160,800,3)):
    model = Sequential()

    input_shape = (time_len, )+img_shape

    model.add(TimeDistributed(Conv2D(df_dim, (8, 8), strides=(5, 5), name="c1", padding="same"), input_shape = input_shape))
    model.add(TimeDistributed(Activation('elu')))  # (32, 160, 64)
    
    model.add(TimeDistributed(Conv2D(df_dim*2, (5, 5), strides=(2, 2), name="c2",  padding="same")))
    #model.add(TimeDistributed(BN(name="e_bn1", epsilon=1e-5, gamma_initializer=mean_normal)))
    model.add(TimeDistributed(Activation('elu')))  # (16, 80, 128)

    model.add(TimeDistributed(Conv2D(df_dim*3, (5, 5), strides=(2, 2), name="c3", padding="same")))
    #model.add(TimeDistributed(BN(name="e_bn2", epsilon=1e-5, gamma_initializer=mean_normal)))
    model.add(TimeDistributed(Activation('elu')))  # (8,40, 192)

    model.add(TimeDistributed(Conv2D(df_dim*4, (5, 5), strides=(2, 2), name="c4", padding="same")))
    #model.add(TimeDistributed(BN(name="e_bn3", epsilon=1e-5, gamma_initializer=mean_normal)))
    model.add(TimeDistributed(Activation('elu')))  # (4,20, 256)

    model.add(TimeDistributed(Conv2D(df_dim*6, (3, 3), strides=(2, 2), name="c5", padding="same")))
    #model.add(TimeDistributed(BN(name="e_bn4", epsilon=1e-5, gamma_initializer=mean_normal)))
    model.add(TimeDistributed(Activation('elu')))  # (2, 10, 384)

    model.add(TimeDistributed(Conv2D(df_dim*8, (3, 3), strides=(2, 2), name="c6", padding="same")))
    #model.add(TimeDistributed(BN(name="e_bn5", epsilon=1e-5, gamma_initializer=mean_normal)))
    model.add(TimeDistributed(Activation('elu')))  # (1, 5, 512)

    model.add(TimeDistributed(Flatten()))
    
    model.add(TimeDistributed(Dense(z_dim, name="e_h3_lin", activation="tanh")))

    model.add(LSTM(return_sequences=True,  units=z_dim*2))
    model.add(Dropout(0.5))

    model.add(LSTM(return_sequences=True, units=z_dim))
    model.add(Dropout(0.5))

    model.add(LSTM(return_sequences=False, units=100))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model

def get_model(sess, image_shape=(160, 800, 3), df_dim=64, batch_size=32, training_mode=True,
              name="cnn_lstm"):

  K.set_session(sess)
  checkpoint_dir = './outputs/results_' + name
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  with tf.variable_scope(name):
    # sizes
    z_dim = 512
    time_len = 5
    learning_rate = 5e-4

    L = cnn_lstm(batch_size, df_dim, z_dim, time_len=time_len)
    L.compile("sgd", "mse")
    print("L.shape: ", L.output_shape)
    #E.trainable = True

    # Network
    target = tf.placeholder(tf.float32, shape=(batch_size, 1))
    Img = Input((time_len,) + image_shape)
    print(Img.shape)
    out = L(Img)

    # costs
    loss = tf.reduce_mean(tf.square(target - out))
    print("CNN_LSTM variables:")
    var_list = [v for v in tf.trainable_variables() \
            if (v.name.startswith('cnn_lstm'))]  
    for v in var_list:
      print(v.name)

    t_optim = tf.train.AdamOptimizer(learning_rate, beta1=.5).minimize(loss, var_list=var_list)
    sess.run(tf.global_variables_initializer())

    # summaries
    sum_loss = tf.summary.scalar("loss", loss)
    sum_loss_test = tf.summary.scalar("loss_test", loss)

    # saver
    if training_mode:
      saver = tf.train.Saver()
      time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
      logdir = './logs/cnn_lstm/'+time+''
      if not os.path.exists(logdir):
        os.makedirs(logdir)
      writer = tf.summary.FileWriter(logdir, sess.graph)

    def train_l(images, labels, counter, sess=sess):
      _, loss_sum, loss_value = sess.run([t_optim, sum_loss, loss], \
              feed_dict={Img: images, target: labels, K.learning_phase(): 1})
      print("###epoch:{0}---loss:{1:.3}".format(counter, loss_value))

      writer.add_summary(loss_sum, counter)

    def test_l(images, labels, counter, sess=sess):
      loss_sum_test = sum_loss_test.eval(feed_dict={Img: images, target: labels, K.learning_phase(): 0})

      writer.add_summary(loss_sum_test, counter)

    def f_load():
      L.load_weights(checkpoint_dir+"/L_weights.keras")
      print("model weights has been loaded")

    def f_save():
      L.save_weights(checkpoint_dir+"/L_weights.keras", True)
      print("model weights has been saved")

    return train_l, test_l, f_save, f_load, L

def batch_gen(it):
    batch = next(it) #get data for training from data server
    batch = list(batch)
    batch_x = batch[0]
    batch_y = batch[1]
    return batch_x, batch_y
if __name__ =="__main__":
  nb_epoch = 40000
  it = gen(20, "localhost", port=5557)
  it_test = gen(20, "localhost", port=5556)
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    l_train, l_test, saver, loader, L = get_model(sess)
    #loader()
    epoch = 0
    while epoch < nb_epoch:
      batch_x, batch_y = batch_gen(it)
      l_train(batch_x, batch_y, epoch)
      epoch = epoch+1
      if epoch % 100 == 0:
        batch_x, batch_y = batch_gen(it_test)
        l_test(batch_x, batch_y, epoch/100)
        saver()