
####################################################
# Cleaning Data because pandas doesnt work for me
####################################################

import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)


def getCleanedData(filename):
    data = pd.read_csv(filename)

    # Diagnosis
    y = data.iloc[:, 1]
    y = y.replace(['M', 'B'], [1, -1])

    # Looking at columns: radius, texture, perimeter, area,
    #   smoothness, compactness, concavity, concave points
    # symmetry and fractal dimension
    data = data.iloc[:, 2:12]

    return data.values, y.values


#  X, y = getCleanedData("data.csv")
