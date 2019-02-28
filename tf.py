import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 下载MNIST数据集到'MNIST_data'文件夹并解压
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 设置权重weights和偏置biases作为优化变量，初始值设为0
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, ys:v_ys, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建模型
with tf.name_scope('Inputs'):
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(xs, [-1, 28, 28, 1])#


## conv1 layer ##
with tf.name_scope('Conv-Pool_1'):
    W_conv1 = weight_variable([5, 5, 1, 32])# patch 5x5
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)                          # output size 14*14*32

with tf.name_scope('Conv-Pool_2'):
    W_conv2 = weight_variable([5, 5, 32, 64])# patch 5x5
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)                          # output size 7*7*64

with tf.name_scope('Func_1'):
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64 ]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('Func_2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('Train'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 开始训练
init = tf.initialize_all_variables()
sess = tf.Session()
merge_op = tf.summary.merge_all()                       # operation to merge all summary
writer = tf.summary.FileWriter('./log', sess.graph)     # write to file
sess.run(init)
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)                                # 每次随机选取100个数据进行训练，即所谓的“随机梯度下降（Stochastic Gradient Descent，SGD）”
    _, result = sess.run([train_step, merge_op], feed_dict={xs: batch_xs, ys:batch_ys, keep_prob:0.5})                  # 正式执行train_step，用feed_dict的数据取代placeholder
    writer.add_summary(result, i)
    if i % 100 == 0:
        # 每训练100次后评估模型
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        # correct_prediction = tf.equal(tf.argmax(, 1), tf.arg_max(y_real, 1))       # 比较预测值和真实值是否一致
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))             # 统计预测正确的个数，取均值得到准确率
        # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels}))
