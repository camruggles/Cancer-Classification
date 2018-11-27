# Input: numpy matrix X of features, with n rows (samples), d columns (features)
#           X[i,j] is the j-th feature of the i-th sample
#        numpy vector y of labels, with n rows (samples), 1 column
#           y[i] is the label (+1 or -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column

import cvxopt as co
import numpy as np

def train(X, y):
    d = len(X[0, :])
    n = len(X[:, 0])
    H = np.identity(d)
    f = np.zeros(d)
    A = np.empty([n, d])
    for i in range(0, n):
        for j in range(0, d):
            A[i, j] = -y[i] * X[i, j]
    b = np.full(n, -1)
    co.solvers.options['show_progress'] = False
    theta = np.array(co.solvers.qp(co.matrix(H,tc='d'),co.matrix(f,tc='d'),co.matrix(A,tc='d'),co.matrix(b,tc='d'))['x'])
    theta = theta.reshape((d, 1))
    return theta

def test(theta, x):
    if np.dot(theta.T, x) > 0:
        label = 1
    else:
        label = -1
    return label
