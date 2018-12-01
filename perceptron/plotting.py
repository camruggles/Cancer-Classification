import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filenames = ['bootstrap5.csv',
             'bootstrap10.csv',
             'bootstrap20.csv']
# 'kfold2.csv',
# 'kfold5.csv',
# 'kfold10.csv',
# 'loocv.csv']
boots = [5, 10, 20]

i = 131
for filename in filenames:
    df = pd.read_csv(filename)
    a = np.array(df)

    a = a[:, 1:]

    print a
    acc = a[0, :]
    err = a[1, :]
    rec = a[2, :]
    pre = a[3, :]
    spec = a[4, :]

    mean_acc = np.mean(acc)
    mean_err = np.mean(err)
    mean_rec = np.mean(rec)
    mean_pre = np.mean(pre)
    mean_spec = np.mean(spec)

    print filename

    names = ['acc', 'err', 'recall', 'prec', 'spec']
    values = [mean_acc, mean_err, mean_rec, mean_pre, mean_spec]
    ax = plt.subplot("{}".format(i))
    ax.set_title(filename)
    ax.bar(names, values)
    i += 1


plt.suptitle("Bootstrapping")
plt.show()
