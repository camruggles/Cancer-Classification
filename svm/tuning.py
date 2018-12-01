import numpy as np
from sklearn import svm, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import read_clean
import matplotlib.pyplot as plt

def bootstrapping_for_tuning(B, X_subset, y_subset, C):
    n = len(X_subset)
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0, n, n))
        test_samples = list(set(range(n)) - set(train_samples))
        # print np.unique(y_subset)
        alg = svm.SVC(kernel = 'linear', C = C)
        alg.fit(X_subset[train_samples], y_subset[train_samples])
        bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
    err = np.mean(bs_err)
    return err

X, y = read_clean.getCleanedData("data.csv")

#choosing a good gamma and C to use - parameter tuning from GridSearchCV
#grid_search = GridSearchCV(svm.SVC(kernel = 'linear'), {'C': [100, 1000, 10000, 100000]}, cv=10)
#grid_search.fit(X, y)
#print grid_search.best_params_
#exit()

#hyperparameter tuning with cross validation (2-fold)
positive_samples = list(np.where(y==1)[0])
negative_samples = list(np.where(y==-1)[0])

# print positive_samples
# print negative_samples

samples_in_fold1 = positive_samples[0:len(positive_samples)/2] + negative_samples[0:len(negative_samples)/2]
samples_in_fold2 = positive_samples[len(positive_samples)/2:] + negative_samples[len(negative_samples)/2:]

# print samples_in_fold1
# print samples_in_fold2

C_list = [100, 1000, 10000]
B = 30

y_pred = np.zeros(len(X))

best_err = 1.1
best_C = 0.0
for C in C_list:
    err = bootstrapping_for_tuning(B, X[samples_in_fold1], y[samples_in_fold1], C)
    print "C=", C, ", err=", err
    if err < best_err:
        best_err = err
        best_C = C

print "best_C=", best_C

alg = svm.SVC(kernel = 'linear', C = best_C)
alg.fit(X[samples_in_fold1], y[samples_in_fold1])
y_pred[samples_in_fold2] = alg.predict(X[samples_in_fold2])

best_err = 1.1
best_C = 0.0
for C in C_list:
    err = bootstrapping_for_tuning(B, X[samples_in_fold2], y[samples_in_fold2], C)
    print "C=", C, ", err=", err
    if err < best_err:
        best_err = err
        best_C = C

print "best_C=", best_C

alg = svm.SVC(kernel = 'linear', C = best_C)
alg.fit(X[samples_in_fold2], y[samples_in_fold2])
y_pred[samples_in_fold1] = alg.predict(X[samples_in_fold1])

err = np.mean(y!=y_pred)
print "Hyperparameter tuning err=", err
