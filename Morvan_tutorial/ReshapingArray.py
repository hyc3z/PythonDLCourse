'''
Two common numpy functions used in deep learning are np.shape and np.reshape().
- X.shape is used to get the shape (dimension) of a matrix/vector X.
- X.reshape(…) is used to reshape X into some other dimension.

For example, in computer science, an image is represented by a 3D array of shape (length,height,depth=3)
. However, when you read an image as the input of an algorithm you convert it to a vector of shape (length∗height∗3,1). In other words, you “unroll”, or reshape, the 3D array into a 1D vector.
---------------------
作者：Koala_Tree
来源：CSDN
原文：https://blog.csdn.net/Koala_Tree/article/details/78057033
版权声明：本文为博主原创文章，转载请附上博文链接！
'''


import numpy as np


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """ ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return v


def main():
    image = np.array([[[ 0.67826139, 0.29380381], [ 0.90714982, 0.52835647], [ 0.4215251 , 0.45017551]], [[ 0.92814219, 0.96677647], [ 0.85304703, 0.52351845], [ 0.19981397, 0.27417313]], [[ 0.60659855, 0.00533165], [ 0.10820313, 0.49978937], [ 0.34144279, 0.94630077]]])

    print(image2vector(image))


if __name__ == '__main__':
    main()
