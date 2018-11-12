# Input: number of iterations L
#   numpy matrix X of features, with n rows (samples), d columns (features)
#       X[i,j] is the j-th feature of the i-th sample
#   numpy vector y of labels, with n rows (samples), 1 column
#       y[i] is the label (+1 or -1) for the i-th sample
# Output: numpy vector theta of d rows, 1 column

import numpy as np


def train(L, X, y):
    n = len(X[0, :])
    w = np.zeros(n)
    for iter in range(1, L+1):
        for i in range(0, len(y)):
            x = X[i, :]
            a = np.dot(w, x)
            if a*y[i] <= 0:
                w = w + y[i]*X[i]

    theta = w
    theta = theta.reshape((n, 1))
    return theta


# Input: numpy vector theta of d rows, 1 column
#     numpy vector x of d rows, 1 column
# Output: label (+1 or -1)


def test(theta, x):
    w = theta
    if np.dot(w.T, x) > 0:
        label = 1
    else:
        label = -1
    return label
