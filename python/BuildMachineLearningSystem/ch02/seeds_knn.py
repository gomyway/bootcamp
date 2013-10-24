# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from load import load_dataset
import numpy as np
from knn import learn_model, apply_model, accuracy

features, labels = load_dataset('seeds')

#labels are like ['1','1','2','3']
label_names = np.array(['Kama', 'Rosa' , 'Canadian'])

labels = label_names[labels.astype(np.int)-1]

def cross_validate(features, labels):
    error = 0.0
    for fold in range(10):
        training = np.ones(len(features), bool)
        training[fold::10] = 0 #set ith element of every 10 to be zero
        testing = ~training
        model = learn_model(1, features[training], labels[training])
        test_error = accuracy(features[testing], labels[testing], model)
        error += test_error

    return error / 10.0

error = cross_validate(features, labels)
print('Ten fold cross-validated error was {0:.1%}.'.format(error))

#normalize the features with z-scoring z = (x-mean)/sigma
features -= features.mean(axis=0) #axis=0 means compute one mean for each column
features /= features.std(axis=0) #axis=1 means compute one mean for each row
#features.mean() will compute one mean for all elements (flattened array)


#above is the same as below:
#for fi in xrange(features.shape[1]):
#    features[:,fi] -= features[:,fi].mean()
#    features[:,fi] /= features[:,fi].std()

error = cross_validate(features, labels)
print('Ten fold cross-validated error after z-scoring was {0:.1%}.'.format(error))
