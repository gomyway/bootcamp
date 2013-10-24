# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np

#knn, k means number of neighbors to check, k>1 will take the majority vote from k neighbors
def learn_model(k, features, labels):
    return k, features.copy(), labels.copy()


def plurality(xs):#return the one with most number of occurrence
    from collections import defaultdict
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    maxv = max(counts.values())
    for k, v in counts.items():
        if v == maxv:
            return k


def apply_model(features, model):
    k, train_feats, labels = model
    results = []
    for f in features:
        label_dist = []
        for t, ell in zip(train_feats, labels):
            label_dist.append((np.linalg.norm(f - t), ell)) #np.linalg.norm(array) = sqrt(sum(x^2))
        label_dist.sort(key=lambda d_ell: d_ell[0]) #sort by distance in ascending order
        label_dist = label_dist[:k] #take the closest k neighbors
        results.append(plurality([ell for _, ell in label_dist]))
    return np.array(results)


def accuracy(features, labels, model):
    preds = apply_model(features, model)
    return np.mean(preds == labels)
