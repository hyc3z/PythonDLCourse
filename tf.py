import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram('weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram('biases', biases)
        with tf.name_scope('Y'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            tf.summary.histogram('wx_plus_b', Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,1], name='x_input')
    ys = tf.placeholder(tf.float32,[None,1], name='y_input')

l1 = add_layer(xs, 1, 10, tf.nn.relu)
l2 = add_layer(l1, 10, 1)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - l2),reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
file_writer = tf.summary.FileWriter('./logs', sess.graph)
summaries = tf.summary.merge_all()
sess.run(init)

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()


for i in range(2000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    summ = sess.run(summaries, feed_dict={xs: x_data, ys: y_data})
    file_writer.add_summary(summ, global_step=i)
    if not i%50 :
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # prediction_value = sess.run(l2, feed_dict={xs:x_data})
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # plt.pause(0.1)
        # plt.ioff()
        # plt.show()

