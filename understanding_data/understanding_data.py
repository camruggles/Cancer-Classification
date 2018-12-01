###################################################
# This will be used for the project introduction
###################################################

import read_clean
import pandas as pd
import numpy as np
from ggplot import *

# Getting the Data
X,y = read_clean.getCleanedData("data.csv")
X = pd.DataFrame(X)
X.columns = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave points', 'symmetry' ,'fractal dimension' ]

features = list(X)

X['diagnosis'] = y

###############################################
# Exploring the Data
###############################################
for f in features:
    print(ggplot(aes(x = f, fill = 'factor(diagnosis)', color = 'factor(diagnosis)'), data=X) + geom_histogram(alpha = 0.6)) + ggtitle(f)