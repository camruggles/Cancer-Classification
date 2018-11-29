import read_clean as dataCollector
import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
sns.set()


def main():
    #  extract the data and the labels
    X, y = dataCollector.getCleanedData("data.csv")
    n, d = X.shape
    print 'n:', n
    print 'd:', d
    u, s, vh = np.linalg.svd(X)
    print s
    # ax = sns.heatmap(X)
    print X[0, :]
    # plt.show()


main()
