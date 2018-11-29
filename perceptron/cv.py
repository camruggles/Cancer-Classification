import sys
import numpy as np
import linperceptron as LP
import read_clean as dataCollector


from random import seed
from random import randrange

#  Split a dataset into k foldss


def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def main():
    try:
        foldcount = int(sys.argv[1])
    except IndexError:
        print 'Please list the number of folds for cross validation'
        print 'as a command line argument, for example : python cv.py 10'
        quit()
    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    results = np.zeros(foldcount)
    seed(1)
    n, d = X.shape

    # create a dataset with the labels and the data mixed together
    dataset = np.zeros((n, d+1))
    dataset[:, 0] = y
    dataset[:, 1:] = X

    # extract 10 folds from the data
    folds = cross_validation_split(dataset, foldcount)
    # print(folds)

    # for each fold, figure out how many data points are in the folds
    #  excluding the one about to be tested
    for i in xrange(foldcount):
        totalRows = 0
        totalCols = 0
        n, d = dataset.shape
        totalCols = d
        for j in xrange(foldcount):
            if j == i:
                continue
            currentFold = np.matrix(folds[j])
            n, d = currentFold.shape
            totalRows += n

        # construct the training set with the row count obtained
        # and fill it with the training data
        trainingSet = np.empty((totalRows, totalCols))
        rowCounter = 0
        for j in xrange(foldcount):
            if j == i:
                continue
            currentfold = np.matrix(folds[j])
            n, d = currentfold.shape
            trainingSet[rowCounter:rowCounter+n, :] = currentfold[j]
            rowCounter += n

        # extract the labels and the data
        y2 = trainingSet[:, 0]
        X2 = trainingSet[:, 1:]

        # train the perceptron
        theta = LP.train(1000, X2, y2)

        # use the last fold for the test set
        # extract the labels and the test points
        testSet = np.matrix(folds[i])
        n, d = testSet.shape
        ttc = 0  # Total Test count
        cc = 0  # Total correct count
        for j in xrange(n):
            # extract the test point and test label
            test_point = testSet[j, 1:].T
            test_label = testSet[j, 0]
            # count if the test was good or not
            if LP.test(theta, test_point) == test_label:
                cc += 1
            ttc += 1

        # print the results of the test
        sys.stdout.write('Fold %d, total correct %d / %d\n' % (i, cc, ttc))
        results[i] = float(cc) / float(ttc)


main()
