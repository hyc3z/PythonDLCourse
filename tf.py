import tensorflow as tf
import numpy as np


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


for _ in range(201):
    sess.run(train)
    if not _%20 :
        print(_, sess.run(Weights), sess.run(biases))

file_writer = tf.summary.FileWriter('./logs', sess.graph)
