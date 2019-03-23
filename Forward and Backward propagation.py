
import numpy as np
from Sigmoid import sigmoid


def propagate(w, b, X, Y):
    """

    :param w: weights, a numpy array size ( num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    :return: cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    """

    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X)+b) #compute activation
    cost = -(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)) # compute cost

    # BACKWARD PROPAGATION(TO FIND GRAD)
    dw = (1.0/m)*np.dot(X, (A-Y).T)
    db = (1.0/m)*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {
        "dw": dw,
        "db": db,
             }

    return grads, cost


w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]),np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))


