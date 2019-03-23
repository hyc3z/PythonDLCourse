# ---------------------
# 作者：Koala_Tree
# 来源：CSDN
# 原文：https://blog.csdn.net/Koala_Tree/article/details/78057033
# 版权声明：本文为博主原创文章，转载请附上博文链接！

import math
import random
import numpy as np


def sigmoid(x):
    """
        Exercise: Build a function that returns the sigmoid of a real number x. Use math.exp(x) for the exponential function.
    #
    # Reminder:
    # sigmoid(x)=11+e−x
    # is sometimes also k# nown as the logistic function. It is a non-linear function used not only in Machine Learning (Logistic Regression), but also in Deep Learning.
    :param x:
    :return:
    """
    return 1.0/(1.0+1/np.exp(x)) #exp: return e raised to the power of x.


def sigmoid_derivative(x):
    """
    Exercise: Implement the function sigmoid_grad() to compute the gradient
    of the sigmoid function with respect to its input x.
    The formula is:
    sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))
    You often code this function in two steps:
    1. Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
    2. Compute σ′(x)=s(1−s)
    :param x:
    :return:
    """
    return sigmoid(x)*(1-sigmoid(x))


def main():
    print(sigmoid(random.random()))
    x = np.array([1, 2, 3, 4, 5])
    print(sigmoid(x))
    print(sigmoid_derivative(x))


if __name__ == '__main__':
    main()
