import numpy as np


def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """ ### START CODE HERE ### (≈ 3 lines of code) # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x) # (n,m) # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True) # (n,1) # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum # (n,m) 广播的作用 ### END CODE HERE ###
    return s


def main():
    x = np.array([[1, 2, 3],[3,4,5]])
    print(softmax(x))


if __name__ == '__main__':
    main()

