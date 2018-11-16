import cvxopt as co
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
# Input: numpy matrix X of featrues, with n rows (samples), d columns (features)
#	X[i, j] is the j-th feature of the i-th sample
#	numpy vector y of labels, with n rows (samples), 1 column
#	y[i] is the label (+1 of -1) of the i-th sample
# Output: numpy vector alpha of n rows, 1 column
def train(X, y):
    d = np.size(X, 1)
    n = np.size(X, 0)
    rbf_svc = svm.SVC(kernel = 'rbf')
    rbf_svc.fit(X, y)
    rbf_svc.predict(X)