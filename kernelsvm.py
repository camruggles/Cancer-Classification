import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import read_clean
import matplotlib.pyplot as plt

###############################################
# Using the scikit learn svm package to create an rbf kernel svm
###############################################

X, y = read_clean.getCleanedData("data.csv")

#choosing a good gamma and C to use 

#creating and fitting the kernel to the data
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.1, C = 1)   #should modify hyperparams to test
rbf_svc.fit(X, y)

#visualizing the data

