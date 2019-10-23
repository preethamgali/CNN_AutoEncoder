import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
import os
from keras.datasets import mnist

def normalize_data(data):
  data[data<=100] = 0
  data[data>100] = 1
  return data 

def reshape_img(data):
    data = normalize_data(data)
    lis = []
    for i in data:
        lis.append(i.reshape((28,28,1)))
    return np.asarray(lis)
    
with tf.device('/CPU:0'):

  # shape = [filter_w, filtr_h, filter_d, number_of_filters]
  conv1_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,1,10],mean =0 ,stddev=1),name = "conv1_filters")
#   conv1_bias = tf.Variable(tf.zeros(10),name="conv1_bias")

  conv2_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,10,10],mean =0 ,stddev=1),name = "conv2_filters")
#   conv2_bias = tf.Variable(tf.zeros(10),name="conv2_bias")

  conv3_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,10,10],mean =0 ,stddev=1),name = "conv3_filters")
#   conv3_bias = tf.Variable(tf.zeros(10),name="conv3_bias")

  uconv3_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,10,10],mean =0 ,stddev=1),name = "uconv3_filters")
#   uconv3_bias = tf.Variable(tf.zeros(10),name="uconv3_bias")

  uconv2_filters = tf.Variable(tf.random.truncated_normal(shape = [3,3,10,1],mean =0 ,stddev=1),name = "uconv2_filters")
#   uconv2_bias = tf.Variable(tf.zeros(1),name="uconv2_bias")

  # 28x28 ---> 28x28x1
  input_img = tf.placeholder(tf.float32,(None,28,28,1),name = 'input')

  # 28x28x1 --> 14x14x5
  conv1 = tf.nn.convolution(input_img , conv1_filters, padding = "SAME", strides = [1,1,1,1],name="conv1")
#   conv1 = tf.nn.bias_add(conv1,conv1_bias,name="conv1_b")
  conv1 = tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding = 'SAME',name = "conv1_maxp")
  conv1 = tf.nn.relu(conv1,name="conv1_relu")

  # 14x14x5 --> 7x7x10
  conv2 = tf.nn.convolution(conv1 , conv2_filters, padding = "SAME", strides = [1,1,1,1],name="conv2")
#   conv2 = tf.nn.bias_add(conv2,conv2_bias,name="conv2_b")
  conv2 = tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding = 'SAME',name = "conv2_maxp")
  conv2 = tf.nn.relu(conv2,name="conv2_relu")

  # 7x7x10
  conv3 = tf.nn.convolution(conv2 , conv3_filters, padding = "SAME", strides = [1,1,1,1],name="conv3")
#   conv3 = tf.nn.bias_add(conv3,conv3_bias,name="conv3_b")


  encode = tf.nn.relu(conv3,name="encode")

  # 7x7x10 --> 14x14x10
  encode = tf.nn.relu(conv3,name="encode")
  uconv3 = tf.image.resize_nearest_neighbor(conv3,size=(14,14),name = "upscale1")
  uconv3 = tf.nn.convolution(uconv3 , uconv3_filters, padding = "SAME", strides = [1,1,1,1],name="uconv3")
#   uconv3 = tf.nn.bias_add(uconv3,uconv3_bias,name="uconv3_b")
  uconv3 = tf.nn.relu(uconv3,name="uconv3_relu")

  # 14x14x10 ---> 28x28x1
  uconv2 = tf.image.resize_nearest_neighbor(uconv3,size=(28,28),name = "upscale2")
  uconv2 = tf.nn.convolution(uconv2 , uconv2_filters, padding = "SAME", strides = [1,1,1,1],name="uconv2")
#   uconv2 = tf.nn.bias_add(uconv2,uconv2_bias,name="uconv2_b")
  uconv2 = tf.nn.sigmoid(uconv2,name="uconv3_sigmoid")

  error = tf.nn.sigmoid_cross_entropy_with_logits(labels = input_img,logits = uconv2)
  t_error = tf.reduce_mean(error)

  train = tf.train.AdamOptimizer(0.01).minimize(t_error)
  
  
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = reshape_img(X_train)
numbers = []
taken = []
for img,j in zip(X_train,y_train):
      if j not in taken:
        numbers.append(img)
      if len(numbers) == 10:
        break
outs = [];errors = []
with tf.Session() as sess:
  epoch = 100
  sess.run(tf.global_variables_initializer())

  for e in range(epoch):
    out = sess.run(train,{input_img:[numbers[1]]})
    if e%10 == 0:
      out = sess.run(uconv2,{input_img:[numbers[1]]})
      er = sess.run(t_error,{input_img:[numbers[1]]})
      outs.append(out[:,:,:,0][0])
      errors.append(er)
