# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from load import load_dataset
import numpy as np
from threshold import learn_model, apply_model, accuracy

features, labels = load_dataset('seeds')
#labels are like ['1','1','2','3']
label_names = np.array(['Kama', 'Rosa' , 'Canadian'])

labels = label_names[labels.astype(np.int)-1]

labels = labels == 'Canadian'

error = 0.0
for fold in range(10):
    training = np.ones(len(features), bool)
    training[fold::10] = 0
    testing = ~training
    model = learn_model(features[training], labels[training])
    test_error = accuracy(features[testing], labels[testing], model)
    error += test_error

error /= 10.0

print('Ten fold cross-validated error was {0:.1%}.'.format(error))
