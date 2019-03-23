"""
Exercise: Implement the numpy vectorized version of the L1 loss. You may find the function abs(x) (absolute value of x) useful.
"""


import numpy as np
import decimal
from decimal import Decimal

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
Returns:
    loss -- the value of the L1 loss function defined above
    """

    loss = np.sum(np.abs(y - yhat))
    return loss


def L2(yhat, y):
    loss = np.sum(np.power((y-yhat), 2))
    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))
print("L2 = " + str(L2(yhat,y)))


