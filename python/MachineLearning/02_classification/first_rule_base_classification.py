# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data['data']
labels = data['target_names'][data['target']]

#setosa is easy to be separated by feature petal length with 100% accuracy
#now we only focus on separating versicolor and virginica, a binary classification problem
setosa = (labels == 'setosa')
features = features[~setosa]
labels = labels[~setosa]
virginica = (labels == 'virginica')


best_acc = -1.0
for fi in range(features.shape[1]): 
#features.shape, dimension of feature space
#Out[1]: (100, 4)
    thresh = np.unique(features[:, fi].copy()) #get unique values for this feature as possible thresholds
    thresh.sort() #sort the thresholds
    for t in thresh:
        pred = (features[:, fi] > t)
        acc = (pred == virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
print('Best cut is {0} on feature {1}, which achieves accuracy of {2:.1%}.'.format(
    best_t, best_fi, best_acc))
