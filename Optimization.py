"""
Write down the optimization function. The goal is to learn w and b by minimizing the cost function J.
For a parameter theta, the uppdate rule is theta = theta - alpha* d(theta), where alpha is the learning rate.
"""

#  GRADED FUNCTION: optimize
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


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient decent algorithm

    Arguments:
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of shape (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop.
    :param learning_rate: learning rate of the gradient decent update rule
    :param print_cost: True to print the loss every 100 steps
    :return:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule
        w = w - learning_rate*dw
        b = b - learning_rate*db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate=0.009, print_cost=False)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))

def predict(w, b, X):
    """

    :param w:
    :param b:
    :param X: data of size (num_px * num_px * 3, number of examples
    :return: Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X

    """

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    assert (Y_prediction.shape == (1,m))

    return Y_prediction

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))
