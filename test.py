import tensorflow as tf


# def weight_varible(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
#
# sess = tf.InteractiveSession()
#
# W_conv1 = weight_varible([2, 4, 5])
#
# sess.run(tf.initialize_all_variables())
#
# print(sess.run(W_conv1))

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')