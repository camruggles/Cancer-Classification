
####################################################
# Cleaning Data because pandas doesnt work for me 
####################################################

import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

def getCleanedData(filename):
    data = pd.read_csv(filename)

    # Diagnosis
    y = data.iloc[:,1]
    y = y.replace(['M','B'], [1, -1])

    # Looking at columns: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points
    # symmetry and fractal dimension 
    data = data.iloc[:, 2:11]
    
    return data.values, y.values
    
X, y = getCleanedData("data.csv")

'''
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
print X

'''