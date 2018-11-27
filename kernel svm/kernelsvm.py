import numpy as np
from sklearn import svm, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import read_clean
import matplotlib.pyplot as plt

###############################################
# Using the scikit learn svm package to create an rbf kernel svm
###############################################

X, y = read_clean.getCleanedData("data.csv")

#choosing a good gamma and C to use - parameter tuning from GridSearchCV
grid_search = GridSearchCV(svm.SVC(kernel = 'rbf'), {'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}, cv=10)

#creating and fitting the kernel to the data
grid_search.fit(X, y)
print grid_search.best_params_
gamma, C = grid_search.best_params_

# rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.1, C = 1)   #should modify hyperparams to test - see above gridsearch for finding best params
# rbf_svc.fit(X, y)

#LOOCV
n = len(X)
d = len(X[0])

y_pred = np.zeros((n, 1))
X, y = read_clean.getCleanedData("data.csv")
y = [0 if x == -1 else x for x in y]

for i in range(n):
    range_except_i = range(i) + range(i+1, n)

    X, y = read_clean.getCleanedData("data.csv")
    y = [0 if x == -1 else x for x in y]

    X_train = X[range_except_i]
    y_train = [y[t] for t in range_except_i]

    rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.1, C = 1)   #should modify hyperparams to test - see above gridsearch for finding best params
    rbf_svc.fit(X, y)

    y_pred[i] = rbf_svc.predict(X[i])

err = np.mean(y!=y_pred)
print(err)


#visualizing the data

