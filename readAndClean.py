

import numpy as np
import pandas as pd


# this uses all 32 columns, and gets the 'M' and 'B' labels
#   and converts them to 1 and -1 for the y vector
# I eliminated one column that had strictly nan values
def getCleanedData(filename):
    data = pd.read_csv(filename)

    n, d = data.shape
    n = n
    d = d-2

    uncleaned = data.values
    noNans = uncleaned[:, 1:32]

    labelled = np.zeros((n, d))
    labelled[:, 1:] = noNans[:, 1:]
    for i in xrange(n):
        if noNans[i, 0] == 'B':
            labelled[i, 0] = 1
        elif noNans[i, 0] == 'M':
            labelled[i, 0] = -1
        else:
            print 'Problem in cleaning data'

    y = labelled[:, 0]
    X = labelled[:, 1:]
    return X, y


X, y = getCleanedData("data.csv")
