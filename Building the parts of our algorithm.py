from Sigmoid import sigmoid
import numpy as np


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
