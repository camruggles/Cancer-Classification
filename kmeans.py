import read_clean
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from ggplot import *


# Getting the Data
X,y = read_clean.getCleanedData("data.csv")

###############################################
# Good Value of K
###############################################

# I used the elbow method to help me find the best value of k. 
#The idea is to choose a small value of k that still has a low SSE, and the elbow usually represents 
#where we start to have diminishing returns by increasing k. 

inertia = {}

for i in range(1,10):
    
    # Getting the Data
    X,y = read_clean.getCleanedData("data.csv")
        
    kmeans = KMeans(n_clusters = i, random_state = 1)
    kmeans = kmeans.fit(X)    
    inertia[i] = kmeans.inertia_
    print(kmeans.inertia_)
    
plt.figure()
plt.plot(list(inertia.keys()), list(inertia.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show() 

###############################################
# Visualizing Cluster data
###############################################

# As shown in the graph below, the elbow of the graph is 2. 
# So, 2 is a good value of K and it makes sense as the tumor is either malignant or benign

elbow = 2
X,y = read_clean.getCleanedData("data.csv")
        
kmeans = KMeans(n_clusters = elbow,  random_state = 3)
kmeans = kmeans.fit(X)  

y_kmeans = kmeans.predict(X)

# Converting X to a dataframe
X_predict_pd = pd.DataFrame(X)
# Appending y
X_predict_pd['diagnosis'] = y_kmeans
    
print(ggplot(aes(x=0 , y=1, color = 'diagnosis'), data= X_predict_pd) + geom_point() + xlab("Dimension 1") + ylab("Dimension 2") + ggtitle("Cluster Data"))

###############################################
# Visualizing Actual data
###############################################

# Malignant is denoted by 1 and Benign as -1 in the original dataset
# Converting benign to 0 

X,y = read_clean.getCleanedData("data.csv")
y = [0 if x == -1 else x for x in y]
    
# Converting X to a dataframe
X_pd = pd.DataFrame(X)
# Appending y
X_pd['diagnosis'] = y
    
print(ggplot(aes(x=0 , y=1, color = 'diagnosis'), data= X_pd) + geom_point() + xlab("Dimension 1") + ylab("Dimension 2") + ggtitle("Actual Data"))

###############################################
# LOOCV
###############################################

n = len(X)
d = len(X[0])

y_pred = np.zeros((n,1))
X,y = read_clean.getCleanedData("data.csv")
y = [0 if x == -1 else x for x in y]

for i in range(n):
    all_except_i = range(i) + range(i+1,n)
    
    # Getting the Data
    X,y = read_clean.getCleanedData("data.csv")
    y = [0 if x == -1 else x for x in y]

    X_train = X[all_except_i]
    y_train = [y[t] for t in all_except_i]
    
    # K Means
    kmeans = KMeans(n_clusters = 2, random_state = 3)
    kmeans = kmeans.fit(X_train) 
    
    y_pred[i] = kmeans.predict([X[i]])

err = np.mean(y!=y_pred)
print(err)

###############################################
# Chance
###############################################


# Malignant is denoted by 1 and Benign as -1 in the original dataset
# Converting benign to 0 

X,y = read_clean.getCleanedData("data.csv")
y = [0 if x == -1 else x for x in y]

total = len(y)
malignant =  sum(y)
benign = total - malignant

print(total, malignant, benign)

# Classifier that predicts Benign always
print(float(malignant)/total)

# Classifier that predicts malignant always
print(float(benign)/total)

# Thus, this is not a useful method as the error is greater than 0.37%




