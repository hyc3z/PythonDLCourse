"""
In deep learning, you deal with very large datasets. Hence, a non-computationally-optimal function can become a huge bottleneck in your algorithm and can result in a model that takes ages to run. To make sure that your code is computationally efficient, you will use vectorization. For example, try to tell the difference between the following implementations of the dot/outer/elementwise product.
---------------------
作者：Koala_Tree
来源：CSDN
原文：https://blog.csdn.net/Koala_Tree/article/details/78057033
版权声明：本文为博主原创文章，转载请附上博文链接！
"""

import time
import numpy as np


x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic_dot_c = time.process_time_ns()
dot = 0
for i in range (len(x1)):
    dot += x1[i]*x2[i]
toc_dot_c = time.process_time_ns()
print(toc_dot_c-tic_dot_c)

tic_outer_c = time.process_time_ns()
outer = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc_outer_c = time.process_time_ns()
print(toc_outer_c-tic_outer_c)

tic_elementwise_c = time.process_time_ns()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc_elementwise_c = time.process_time_ns()
print(toc_elementwise_c-tic_elementwise_c)


W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time_ns()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i, j] * x1[j]
toc = time.process_time_ns()
print(toc-tic)

tic = time.process_time_ns()
dot = np.dot(x1,x2)
toc = time.process_time_ns()
print(toc-tic)

tic = time.process_time_ns()
outer = np.outer(x1,x2)
toc = time.process_time_ns()
print(toc-tic)

tic = time.process_time_ns()
mul = np.multiply(x1,x2)
toc = time.process_time_ns()
print(toc-tic)

tic = time.process_time_ns()
dot = np.dot(W, x1)
toc = time.process_time_ns()
print(toc-tic)

