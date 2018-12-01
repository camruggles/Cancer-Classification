import numpy as np
from sklearn import svm, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import read_clean
import matplotlib.pyplot as plt
import sys

#LOOCV
X, y = read_clean.getCleanedData("data.csv")
n = len(X)
d = len(X[0])
C = int(sys.argv[1])

y_pred = np.zeros(n)
y = [0 if x == -1 else x for x in y]

for i in range(n):
    range_except_i = range(i) + range(i+1, n)

    X, y = read_clean.getCleanedData("data.csv")
    y = [0 if x == -1 else x for x in y]

    X_train = X[range_except_i]
    y_train = [y[t] for t in range_except_i]

    svc = svm.SVC(kernel = 'linear', C=C)
    svc.fit(X, y)

    y_pred[i] = svc.predict(X[i].reshape(1, -1))

err = np.mean(y!=y_pred)
print "LOOCV err=", err
