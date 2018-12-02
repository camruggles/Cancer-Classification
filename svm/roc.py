import numpy as np
from sklearn import svm, model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import read_clean
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sys    

X, y = read_clean.getCleanedData("data.csv")
positive_samples = list(np.where(y==1)[0])
negative_samples = list(np.where(y==-1)[0])

samples_in_fold1 = positive_samples[0:len(positive_samples)/2] + negative_samples[0:len(negative_samples)/2]
samples_in_fold2 = positive_samples[len(positive_samples)/2:] + negative_samples[len(negative_samples)/2:]

trainX = X[samples_in_fold1]
trainY = y[samples_in_fold1]
testX = X[samples_in_fold2]
testY = y[samples_in_fold2]

alg = svm.SVC(kernel = 'linear', C = int(sys.argv[1]), probability = True)
alg.fit(trainX, trainY)

probs = alg.predict_proba(testX)
probs = probs[:, 1]

auc = roc_auc_score(testY, probs)

fpr, tpr, thresholds = roc_curve(testY, probs)

plt.figure()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Linear SVM Classifier ROC')
plt.plot(fpr, tpr, color='blue', lw=2, label='Linear SVM ROC area = %0.2f)' % auc)
plt.legend(loc="lower right")
plt.show()
