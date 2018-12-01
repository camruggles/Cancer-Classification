import sys
import numpy as np
import linperceptron as LP
import read_clean as dataCollector

from random import shuffle

#  Split a dataset into k foldss


def cross_validation_split(y, folds=3):
    y_positive = list(np.where(y == 1)[0])
    y_negative = list(np.where(y == -1)[0])
    n_p = int(len(y_positive)/folds)
    n_n = int(len(y_negative)/folds)

    print len(y_positive)
    print len(y_negative)

    print n_p
    print n_n

    shuffle(y_negative)
    shuffle(y_positive)

    split = []
    for i in xrange(folds):
        split.append([])
    k = folds
    for i in range(0, k-1):
        print 'range'
        print (i)*n_p
        print (i+1)*n_p
        print ''
        split[i] += y_positive[i*n_p: (i+1)*n_p]
        split[i] += y_negative[i*n_n: (i+1)*n_n]
    print 'range'
    print (k-1)*n_p
    print (k-1)*n_n
    print ''
    split[k-1] += y_positive[(k-1)*n_p:]
    split[k-1] += y_negative[(k-1)*n_n:]

    print split

    return split


def main():
    try:
        foldcount = int(sys.argv[1])
    except IndexError:
        print 'Please list the number of folds for cross validation'
        print 'as a command line argument, for example : python cv.py 10'
        quit()
    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    accuracy = np.zeros(foldcount)
    precision = np.zeros(foldcount)
    recall = np.zeros(foldcount)
    specificity = np.zeros(foldcount)
    n, d = X.shape

    # extract 10 folds from the data
    split = cross_validation_split(y, foldcount)
    # print(folds)
    print n
    print split

    # for each fold, figure out how many data points are in the folds
    #  excluding the one about to be tested
    for j in xrange(foldcount):
        trainInd = []
        testInd = split[j]
        for i in xrange(foldcount):
            if j == i:
                continue
            trainInd += split[i]

        # construct the training set with the row count obtained
        # and fill it with the training data
        print 'testing fold: ', j
        print 'train indices'
        print trainInd
        print 'test indices'
        print testInd
        trainSet = X[trainInd]
        trainLabels = y[trainInd]

        testSet = X[testInd]
        testLabels = y[testInd]

        print 'ylabels'
        print testLabels

        theta = LP.train(1000, trainSet, trainLabels)

        n = len(testInd)
        # Matt is terrible
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in xrange(n):
            # extract the test point and test label
            test_point = testSet[i]
            test_label = testLabels[i]
            # count if the test was good or not
            testResult = LP.test(theta, test_point)
            print 'labels'
            print testResult
            print test_label

            if testResult == 1 and test_label == 1:
                tp += 1
            if testResult == 1 and test_label == -1:
                fp += 1
            if testResult == -1 and test_label == 1:
                fn += 1
            if testResult == -1 and test_label == -1:
                tn += 1

        # print the results of the test
        print 'tp, fp, fn, tn'
        print tp, fp, fn, tn

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
    print 'accuracy'
    print accuracy
    print 'error'
    print error
    print 'recall'
    print recall
    print 'precision'
    print precision
    print 'specificity'
    print specificity

    print np.mean(accuracy)


main()
