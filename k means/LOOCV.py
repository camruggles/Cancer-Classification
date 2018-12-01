import numpy as np
import read_clean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ggplot import *

n = len(X)
d = len(X[0])

y_pred = np.zeros((n,1))
X,y = read_clean.getCleanedData("data.csv")

for i in range(n):
    all_except_i = range(i) + range(i+1,n)
    
    # Getting the Data
    X_train = X[all_except_i]
    y_train = [y[t] for t in all_except_i]
    
    # K Means
    kmeans = KMeans(n_clusters = 2, random_state = 3)
    kmeans = kmeans.fit(X_train) 
    
    testB = [[   6.981  ,   13.43   ,   43.79   ,  143.5    ,    0.117  ,
          0.07568,    0.     ,    0.     ,    0.193  ,    0.07818]]
        
    testM = [[   28.11   ,    18.47   ,   188.5    ,  2499.     ,     0.1142 ,
           0.1516 ,     0.3201 ,     0.1595 ,     0.1648 ,     0.05525]]
        
    benign = kmeans.predict(testB)
    malignant = kmeans.predict(testM)
    
    val = kmeans.predict([X[i]])
    
    if val == benign:
        y_pred[i] = -1
    elif val == malignant:
        y_pred[i] = 1

count = 0
for i in range(len(y)):
    if y[i] == y_pred[i]:
        count = count + 1
        
print(count/float(len(y)))