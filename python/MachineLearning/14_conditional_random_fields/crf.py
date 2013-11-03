#!/usr/bin/env python
"""
Conditional Random Fields (CRF)
"""

import cPickle as pickle
from collections import defaultdict
from numpy import empty, zeros, ones, log, exp, sqrt, add, int32, abs


def logsumexp(a):
    """
    Compute the log of the sum of exponentials of an array ``a``, :math:`\log(\exp(a_0) + \exp(a_1) + ...)`
    """
    b = a.max()
    return b + log((exp(a-b)).sum())


class CRF(object):
    """
    Conditional Random Field (CRF) for linear-chain structured models.
    """

    def __init__(self, K, M):
        self.W = zeros(M)   # weight vector will go here once we allocate it
        self.K = K          # label domain size; filled in when domain is frozen.

    def save(self, filename):
        """ Saving model to ``filename``. """
        pickle.dump(self, file(filename, 'wb'))

    @classmethod
    def load(cls, filename):
        """ Resurrect a saved ``CRF``. """
        return pickle.load(file(filename, 'r'))

    def log_potentials(self, x):
        """
        Calculate a potential tables for model given current parameters.
        Note: indexing of g is off-by-one
        """
        N = x.N; K = self.K; W = self.W; f = x.feature_table
        g0 = empty(K)
        g = empty((N-1,K,K))
        for y in xrange(K):
            g0[y] = W[f[0,None,y]].sum()
        for t in xrange(1,N):
            for y in xrange(K):
                for yp in xrange(K):
                    g[t-1,yp,y] = W[f[t,yp,y]].sum()
        return (g0, g)

    def __call__(self, x):
        """ Infer the most likely labeling of ``x``. """
        self.preprocess([x])
        return self.argmax(x)

    def argmax(self, x):
        """
        Find the most likely assignment to labels given parameters using the
        Viterbi algorithm.
        """
        N = x.N
        K = self.K
        (g0, g) = self.log_potentials(x)
        B = ones((N,K), dtype=int32) * -1
        # compute max-marginals and backtrace matrix
        V = g0
        for t in xrange(1,N):
            U = empty(K)
            for y in xrange(K):
                w = V + g[t-1,:,y]
                B[t,y] = b = w.argmax()
                U[y] = w[b]
            V = U
        # extract the best path by brack-tracking
        y = V.argmax()
        trace = []
        for t in reversed(xrange(N)):
            trace.append(y)
            y = B[t, y]
        trace.reverse()
        return trace

    def likelihood(self, x):
        """ log-likelihood of ``x`` under model. """
        N = x.N; K = self.K; W = self.W
        (g0, g) = self.log_potentials(x)
        a = self.forward(g0,g,N,K)
        logZ = logsumexp(a[N-1,:])
        return sum(W[k] for k in x.target_features) - logZ

    def forward(self, g0, g, N, K):
        """
        Calculate matrix of forward unnormalized log-probabilities.

        a[i,y] log of the sum of scores of all sequences from 0 to i where
        the label at position i is y.
        """
        a = zeros((N,K))
        a[0,:] = g0
        for t in xrange(1,N):
            ayp = a[t-1,:]
            for y in xrange(K):
                a[t,y] = logsumexp(ayp + g[t-1,:,y])
        return a

    def backward(self, g, N, K):
        """ Calculate matrix of backward unnormalized log-probabilities. """
        b = zeros((N,K))
        for t in reversed(xrange(0,N-1)):
            by = b[t+1,:]
            for yp in xrange(K):
                b[t,yp] = logsumexp(by + g[t,yp,:])
        return b

    def expectation(self, x):
        """
        Expectation of the sufficient statistics given ``x`` and current
        parameter settings.
        """
        N = x.N; K = self.K; f = x.feature_table
        (g0, g) = self.log_potentials(x)

        a = self.forward(g0,g,N,K)
        b = self.backward(g,N,K)

        # log-normalizing constant
        logZ = logsumexp(a[N-1,:])

        E = defaultdict(float)

        # The first factor needs to be special case'd
        # E[ f( y_0 ) ] = p(y_0 | y_[1:N], x) * f(y_0)
        c = exp(g0 + b[0,:] - logZ).clip(0.0, 1.0)
        for y in xrange(K):
            p = c[y]
            if p < 1e-40: continue   # skip really small updates.
            for k in f[0, None, y]:
                E[k] += p

        for t in xrange(1,N):
            # vectorized computation of the marginal for this transition factor
            c = exp((add.outer(a[t-1,:], b[t,:]) + g[t-1,:,:] - logZ)).clip(0.0, 1.0)

            for yp in xrange(K):
                for y in xrange(K):
                    # we can also use the following to compute ``p`` but its quite
                    # a bit slower than the computation of vectorized quantity ``c``.
                    #p = exp(a[t-1,yp] + g[t-1,yp,y] + b[t,y] - logZ).clip(0.0, 1.0)
                    p = c[yp, y]
                    if p < 1e-40: continue   # skip really small updates.
                    # expectation of this factor is p*f(t, yp, y)
                    for k in f[t, yp, y]:
                        E[k] += p

        return E

    def path_features(self, x, y):
        """
        Features of the assignment ``y`` to instance ``x``.
        Note: ``y`` should be a sequence of ``int``.
        """
        F = x.feature_table
        f = list(F[0, None, y[0]])
        f.extend(k for t in xrange(1, x.N) for k in F[t, y[t-1], y[t]])
        return f

    def preprocess(self, data):
        """
        Hook for doing "stuff" to ``data`` before processing it.
        """
        pass

    def sgd(self, data, iterations=20, a0=10, validate=None):
        """ Parameter estimation with stochastic gradient descent (sgd). """
        self.preprocess(data)
        W = self.W
        for i in xrange(iterations):
            print 'Iteration', i
            rate = a0 / (sqrt(i) + 1)
            for x in data:
                for k, v in self.expectation(x).iteritems():
                    W[k] -= rate*v
                for k in x.target_features:
                    W[k] += rate
            if validate:
                validate(self, i)

    def perceptron(self, data, rate=0.01, iterations=20, validate=None):
        """ Parameter estimation with the perceptron algorithm. """
        self.preprocess(data)
        W = self.W
        for i in xrange(iterations):
            print 'Iteration', i
            for x in data:
                for k in self.path_features(x, self.argmax(x)):
                    W[k] -= rate
                for k in x.target_features:
                    W[k] += rate
            if validate:
                validate(self, i)

#    def sgd_l1(self, data, iterations=20, lmd=0.01, eta=0.5, validate=None):
#        """ SGD + FOBOS L1 """
#        self.preprocess(data)
#        W = self.W
#        for i in xrange(iterations):
#            print 'Iteration', i
#            lmd_eta = lmd * eta
#            for x in data:
#
#                for k, v in self.expectation(x).iteritems():
#                    W[k] -= eta*v
#                for k in x.target_features:
#                    W[k] += eta
#                
#                self.W = (self.W > lmd_eta) * (self.W - lmd_eta) + (self.W < -lmd_eta) * (self.W + lmd_eta)
#
#            if validate:
#                validate(self, i)
#
#            eta *= 0.95

#    def bfgs(self):
#        from scipy import optimize
#        likelihood = lambda x:-self.likelihood(fvs, x)
#        likelihood_deriv = lambda x:-self.gradient_likelihood(fvs, x)
#        return optimize.fmin_bfgs(likelihood, theta, fprime=likelihood_deriv)