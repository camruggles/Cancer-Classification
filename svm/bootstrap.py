import numpy as np
import read_clean as dataCollector
import svm
import sys


def bootstrapping(B, X, y, C):

    accuracy = np.zeros(B)
    precision = np.zeros(B)
    recall = np.zeros(B)
    specificity = np.zeros(B)

    n, d = X.shape
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0, n, n))
        test_samples = list(set(range(n)) - set(train_samples))

        # train the model
        theta = svm.train(X[train_samples], y[train_samples], C)

        testSet = X[test_samples]
        testLabels = y[test_samples]
        n2, d2 = testSet.shape

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for j in xrange(n2):
            # extract the test point and test label
            test_point = testSet[j, :].T
            test_label = testLabels[j]
            # count if the test was good or not

            # test the model
            testResult = svm.test(theta, test_point)

            if testResult == 1 and test_label == 1:
                tp += 1
            if testResult == 1 and test_label == -1:
                fp += 1
            if testResult == -1 and test_label == 1:
                fn += 1
            if testResult == -1 and test_label == -1:
                tn += 1

        #print 'tp, tn, fp, fn'
        #print tp, tn, fp, fn
        #print ''

        try:
            accuracy[b] = float(tp + tn) / float(fn + fp + tp + tn)
        except ZeroDivisionError:
            accuracy[b] = 0.0

        try:
            recall[b] = float(tp) / float(tp+fn)
        except ZeroDivisionError:
            recall[b] = 0.0

        try:
            precision[b] = float(tp) / float(tp+fp)
        except ZeroDivisionError:
            precision[b] = 0.0

        try:
            specificity[b] = float(tn) / float(tn+fp)
        except ZeroDivisionError:
            specificity[b] = 0.0

        error = np.ones(B)
        error -= accuracy

    return accuracy, error, recall, precision, specificity

    return bs_err


def main():
    try:
        B = int(sys.argv[1])
        C = int(sys.argv[2])
    except IndexError:
        print 'Please list the number of bootstraps as a cmd line arg'
        print 'for example : python bootstrap.py 10'
        quit()

    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    n, d = X.shape

    # create a dataset with the labels and the data mixed together
    acc, err, recall, precision, specificity = bootstrapping(B, X, y, C)
    print "Using", str(B), "bootstaps, and C =", str(C), "\n"
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


main()
