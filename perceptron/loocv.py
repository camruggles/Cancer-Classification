import numpy as np
import linperceptron as LP
import read_clean as dataCollector

# from random import shuffle


def loocv(X, y):
    n, d = X.shape

    accuracy = np.zeros(n)
    precision = np.zeros(n)
    recall = np.zeros(n)
    specificity = np.zeros(n)

    # running k fold x validation
    for j in xrange(n):

        # breaking up the folds into train and test
        trainInd = []
        testInd = [j]
        for i in xrange(n):
            if j == i:
                continue
            trainInd += [i]

        # construct the training and testing sets

        trainSet = X[trainInd]
        trainLabels = y[trainInd]

        testSet = X[testInd]
        testLabels = y[testInd]

        # train the model
        theta = LP.train(1000, trainSet, trainLabels)

        # Matt is terrible

        # getting information on the statistical results
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        # extract the test point and test label
        test_point = testSet[0]
        test_label = testLabels[0]
        # count if the test was good or not

        # test the model
        testResult = LP.test(theta, test_point)

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
        # print 'tp, tn, fp, fn'
        # print tp, tn, fp, fn
        # print ''

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

    error = np.ones(n)
    error -= accuracy

    return accuracy, error, recall, precision, specificity


def main():

    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    n, d = X.shape

    # initializing output labels
    acc, err, recall, precision, specificity = loocv(X, y)

    # print 'accuracy'
    # print acc
    # print 'error'
    # print err
    # print 'recall'
    # print recall
    # print 'precision'
    # print precision
    # print 'specificity'
    # print specificity

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

    output = [acc, err, recall, precision, specificity]
    import pandas as pd
    df = pd.DataFrame(output)
    df.to_csv("loocv.csv")


main()
