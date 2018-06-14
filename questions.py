__author__ = 'chapter'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):                                                                            # how to understand the layout of 4d array notation
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')                          # why the strides are four dimentional? why is everything four dimentional? BATCH CHANNEL HEIGHT WIDTH

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')       #----------tf.nn.max_pool


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)                               # what is the structure of MNIST?
print("Download Done!")

sess = tf.InteractiveSession()
# --------------------------------------------------------------------------------------
# paras
W_conv1 = weight_varible([5, 5, 1, 32])                                                      #does 32 means 32 different kernels/filters/weights?
b_conv1 = bias_variable([32])

# conv layer-1 -----------------------------------------------------------------------
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])                                                     #-1, infer the shape??

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# ------------------------------------------------------------------------------------
# conv layer-2 -----------------------------------------------------------------------
W_conv2 = weight_varible([5, 5, 32, 64])                                                      #64 different filters?
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)                                      #---------tf.nn.relu
h_pool2 = max_pool_2x2(h_conv2)


# ---------------------------------------------------------------------------------------
# full connection -----------------------------------------------------------------------
W_fc1 = weight_varible([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])                                           # how is the reshape matrix organized?
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout -----------------------------------------------------------------------               #is dropout random? any optimization on dropout layer?
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax -----------------------------------------------------------------------  #what doesj softmax do?
W_fc2 = weight_varible([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 10])



