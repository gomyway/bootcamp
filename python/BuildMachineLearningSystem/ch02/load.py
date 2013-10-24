# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
import os
import scipy as sp

def load_dataset(dataset_name):
    '''
    data,labels = load_dataset(dataset_name)

    Load a given dataset

    Returns
    -------
    data : numpy ndarray
    labels : list of str
    '''
    data = []
    labels = []
    with open('../data/{0}.tsv'.format(dataset_name)) as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            try:
                data.append([float(tk) for tk in tokens[:-1]])
                labels.append(tokens[-1])
            except ValueError: print "Error processling line: %s"%line
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def load_dataset2(dataset_name):
    try:
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
    except NameError: data_dir = "../data"
    data = sp.genfromtxt(os.path.join(data_dir, "{0}.tsv".format(dataset_name)), delimiter="\t") #schema 'features label'
    print(data[:10]) #print first 10 row for a peek of the data
    
    # all examples will have three classes in this file
    
    x = data[:, 0] # take first column
    y = data[:, -1] #take last column
    print("Number of invalid entries:", sp.sum(sp.isnan(y))) #value being nan in y
    #clean the data, remove rows with nan value 
    x = x[~sp.isnan(y)]
    y = y[~sp.isnan(y)]

if __name__ == '__main__':
    import unittest

    class TestLoad(unittest.TestCase):
        def setUp(self):
            pass
        
        def load_dataset(self):
            features, labels = load_dataset('seeds')
        
    unittest.main()  