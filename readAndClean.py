

import numpy as np
import pandas as pd


#this uses all 32 columns, and gets the 'M' and 'B' labels and converts them to 1 and -1 for the y vector
#I eliminated one column that had strictly nan values
def getCleanedData(filename):
	data = pd.read_csv(filename)

	n,d = data.shape
	n = n
	d = d-2

	v1 = data.values
	v2 = v1[:, 1:32]

	v3 = np.zeros((n,d))
	v3[:, 1:] = v2[:, 1:]
	for i in xrange(n):
	    if v2[i, 0] == 'B':
	        v3[i,0] = 1
	    elif v2[i,0] == 'M':
	        v3[i,0] = -1
	    else:
	        print 'Problem in cleaning data'

	y = v3[:, 0]
	X = v3[:, 1:]
	#print v3[0,:]
	#print X[0,:]
	#print y
	return X,y



X,y = getCleanedData("data.csv")