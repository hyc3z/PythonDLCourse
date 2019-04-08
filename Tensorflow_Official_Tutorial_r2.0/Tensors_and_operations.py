import tensorflow as tf


print(tf.add(1,2))
print(tf.reduce_sum([[1, 2], [3, 4]]))
x = tf.constant([[1, 1, 1], [1, 1, 1]])
print(tf.reduce_sum(x))  # 6
print(tf.reduce_sum(x, 0))  # [2, 2, 2]
print(tf.reduce_sum(x, 1))  # [3, 3]
print(tf.reduce_sum(x, 1, keepdims=True))  # [[3], [3]]
print(tf.reduce_sum(x, [0, 1]))  # 6


print(tf.test.is_gpu_available())
import time


def time_matmul(x):
    start = time.time()
    for loop in range(100):
        tf.matmul(x, x)
    result = time.time() - start
    print("10 loops: {:0.2f}ms".format(1000 * result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random.uniform([10000, 10000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

