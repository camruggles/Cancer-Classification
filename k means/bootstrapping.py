import numpy as np
import read_clean as dataCollector
from sklearn.cluster import KMeans

def bootstrapping(B, X, y):
    
    n, d = X.shape
    bs_err = np.zeros(B)

    for b in range(B):
        train_samples = list(np.random.randint(0, n, n))
        test_samples = list(set(range(n)) - set(train_samples))
        
        # K Means - Training
        kmeans = KMeans(n_clusters = 2)
        kmeans = kmeans.fit(X[train_samples]) 
    
        # K Means - Testing
        testSet = X[test_samples]
        testLabels = y[test_samples]
        n2, d2 = testSet.shape
        ttc = 0
        cc = 0
        
        testB = [[   6.981  ,   13.43   ,   43.79   ,  143.5    ,    0.117  ,
          0.07568,    0.     ,    0.     ,    0.193  ,    0.07818]]
        
        testM = [[   28.11   ,    18.47   ,   188.5    ,  2499.     ,     0.1142 ,
           0.1516 ,     0.3201 ,     0.1595 ,     0.1648 ,     0.05525]]
        
        benign = kmeans.predict(testB)
        malignant = kmeans.predict(testM)
        
        for j in xrange(n2):
            # extract the test point and test label
            test_point = testSet[j, :].T
            test_label = testLabels[j]
            # count if the test was good or not
            val = kmeans.predict(test_point.reshape(1, -1))
            
            if val == benign:
                val = -1
            elif val == malignant:
                val = 1
            
            if val == test_label:
                cc += 1
            ttc += 1
        bs_err[b] = float(cc)/float(ttc)
    return bs_err


def main():
    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    n, d = X.shape

    # create a dataset with the labels and the data mixed together
    bs_correct = bootstrapping(5, X, y)
    print 'B = 5 individual correctness'
    print bs_correct
    print 'B = 5 average correctness'
    print np.mean(bs_correct)
    
    
    # create a dataset with the labels and the data mixed together
    bs_correct = bootstrapping(10, X, y)
    print 'B = 10 individual correctness'
    print bs_correct
    print 'B = 10 average correctness'
    print np.mean(bs_correct)
    
    # create a dataset with the labels and the data mixed together
    bs_correct = bootstrapping(20, X, y)
    print 'B = 20 individual correctness'
    print bs_correct
    print 'B = 20 average correctness'
    print np.mean(bs_correct)


main()