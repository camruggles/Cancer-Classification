import numpy as np
import read_clean as dataCollector
import linperceptron as LP


def bootstrapping(B, X, y):
    n, d = X.shape
    bs_err = np.zeros(B)
    for b in range(B):
        train_samples = list(np.random.randint(0, n, n))
        test_samples = list(set(range(n)) - set(train_samples))
        theta = LP.train(1000, X[train_samples], y[train_samples])

        testSet = X[test_samples]
        testLabels = y[test_samples]
        n2, d2 = testSet.shape
        ttc = 0
        cc = 0
        for j in xrange(n2):
            # extract the test point and test label
            test_point = testSet[j, :].T
            test_label = testLabels[j]
            # count if the test was good or not
            if LP.test(theta, test_point) == test_label:
                cc += 1
            ttc += 1
        bs_err[b] = float(cc)/float(ttc)
    return bs_err


def main():
    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    n, d = X.shape

    # create a dataset with the labels and the data mixed together
    dataset = np.zeros((n, d+1))
    dataset[:, 0] = y
    dataset[:, 1:] = X
    bs_correct = bootstrapping(10, X, y)
    print 'individual correctness'
    print bs_correct
    print 'average correctness'
    print np.mean(bs_correct)


main()
