import numpy as np
from sklearn.svm import SVC

def train(X, y, C):
    theta = SVC(C=C, kernel='linear')
    theta.fit(X, y)
    return theta

def test(theta, x):
    x = x.reshape(1, -1)
    return theta.predict(x)
