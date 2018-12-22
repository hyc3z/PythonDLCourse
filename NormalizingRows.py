"""
Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to x∥x∥

(dividing each row vector of x by its norm).

For example, if
x=[023644](3)
then
∥x∥=np.linalg.norm(x,axis=1,keepdims=True)=[556−−√](4)
and
x_normalized=x∥x∥=[0256√35656√45456√](5)
Note that you can divide matrices of different sizes and it works fine: this is called broadcasting and you’re going to learn about it in part 5.

Exercise: Implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).
---------------------
作者：Koala_Tree
来源：CSDN
原文：https://blog.csdn.net/Koala_Tree/article/details/78057033
版权声明：本文为博主原创文章，转载请附上博文链接！
"""