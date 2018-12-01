import numpy as np
import pandas as pd
from ggplot import *

import sys
import read_clean as dataCollector
from sklearn.cluster import KMeans


from random import shuffle, seed

#  Split a dataset into k foldss


def cross_validation_split(y, folds=3):
    seed(1)
    # breaking up the labels into positive labels and negative labels to
    # evenly distribute them
    y_positive = list(np.where(y == 1)[0])
    y_negative = list(np.where(y == -1)[0])

    # getting the size of the respective arrays
    # n_p is number of positives
    # n_n is number of negatives
    n_p = int(len(y_positive)/folds)
    n_n = int(len(y_negative)/folds)

    shuffle(y_negative)
    shuffle(y_positive)

    # creating a 2d array
    split = []
    for i in xrange(folds):
        split.append([])

    k = folds

    # filling the ten folds with indices such that positives and
    # negatives are evenly distributed
    for i in range(0, k-1):
        split[i] += y_positive[i*n_p: (i+1)*n_p]
        split[i] += y_negative[i*n_n: (i+1)*n_n]

    split[k-1] += y_positive[(k-1)*n_p:]
    split[k-1] += y_negative[(k-1)*n_n:]

    return split


def cross_validation(X, y, foldcount):

    accuracy = np.zeros(foldcount)
    precision = np.zeros(foldcount)
    recall = np.zeros(foldcount)
    specificity = np.zeros(foldcount)
    n, d = X.shape

    # extract k folds from the data
    split = cross_validation_split(y, foldcount)

    # running k fold x validation
    for j in xrange(foldcount):

        # breaking up the folds into train and test
        trainInd = []
        testInd = split[j]
        for i in xrange(foldcount):
            if j == i:
                continue
            trainInd += split[i]

        # construct the training and testing sets
        trainSet = X[trainInd]
        trainLabels = y[trainInd]

        testSet = X[testInd]
        testLabels = y[testInd]

        # K Means - Training
        kmeans = KMeans(n_clusters = 2)
        kmeans = kmeans.fit(trainSet) 

        n = len(testInd)
        # Matt is terrible
        
        testB = [[   6.981  ,   13.43   ,   43.79   ,  143.5    ,    0.117  ,
          0.07568,    0.     ,    0.     ,    0.193  ,    0.07818]]
        
        testM = [[   28.11   ,    18.47   ,   188.5    ,  2499.     ,     0.1142 ,
           0.1516 ,     0.3201 ,     0.1595 ,     0.1648 ,     0.05525]]
        
        benign = kmeans.predict(testB)
        malignant = kmeans.predict(testM)

        # getting information on the statistical results
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        
        y_kmeans = kmeans.predict(testSet)

        # Converting X to a dataframe
        X_predict_pd = pd.DataFrame(testSet)
        # Appending y
        X_predict_pd['diagnosis'] = y_kmeans
    
        print(ggplot(aes(x=0 , y=1, color = 'diagnosis'), data= X_predict_pd) + geom_point() + xlab("Dimension 1") + ylab("Dimension 2") + ggtitle("Cluster Data"))

        for i in xrange(n):
            # extract the test point and test label
            test_point = testSet[i]
            test_label = testLabels[i]
            # count if the test was good or not

            # test the model
            val = kmeans.predict(test_point.reshape(1, -1))
            
            if val == benign:
                val = -1
            elif val == malignant:
                val = 1
                
            testResult = val
                
            if testResult == 1 and test_label == 1:
                tp += 1
            if testResult == 1 and test_label == -1:
                fp += 1
            if testResult == -1 and test_label == 1:
                fn += 1
            if testResult == -1 and test_label == -1:
                tn += 1

        # making sure there are no zero denominators
        # probably unnecessary but just in case
        print 'tp, tn, fp, fn'
        print tp, tn, fp, fn
        print ''

        try:
            accuracy[j] = float(tp + tn) / float(fn + fp + tp + tn)
        except ZeroDivisionError:
            accuracy[j] = 0.0

        try:
            recall[j] = float(tp) / float(tp+fn)
        except ZeroDivisionError:
            recall[j] = 0.0

        try:
            precision[j] = float(tp) / float(tp+fp)
        except ZeroDivisionError:
            precision[j] = 0.0

        try:
            specificity[j] = float(tn) / float(tn+fp)
        except ZeroDivisionError:
            specificity[j] = 0.0

        error = np.ones(foldcount)
        error -= accuracy

    return accuracy, error, recall, precision, specificity


def k_fold(m):

    try:
        folds = m
    except IndexError:
        print 'Please list the number of folds for cross validation'
        print 'as a command line argument, for example : python cv.py 10'
        quit()

    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    # initializing output labels
    acc, err, recall, precision, specificity = cross_validation(X, y, folds)

    print 'accuracy'
    print acc
    print 'error'
    print err
    print 'recall'
    print recall
    print 'precision'
    print precision
    print 'specificity'
    print specificity

    print 'mean accuracy'
    print np.mean(acc)
    print 'mean error'
    print np.mean(err)
    print 'mean recall'
    print np.mean(recall)
    print 'mean precision'
    print np.mean(precision)
    print 'mean specificity'
    print np.mean(specificity)

print("K FOLD FOR K = 2")
k_fold(2)
print("K FOLD FOR K = 5")
k_fold(5)
print("K FOLD FOR K = 10")
k_fold(10)