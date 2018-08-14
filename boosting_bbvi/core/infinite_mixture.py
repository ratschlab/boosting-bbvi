#!/usr/bin/python

import numpy as np

class InfiniteMixtureScipy:
    '''
    Set self.params and self.weights on your own. No magic.

    Note that params is a tuple (theta_1, theta_2, ...) where order matters.
    Namely, order must match the argument order of `self.base`.
    '''
    def __init__(self, base):
        assert base.__module__.split('.')[:2] == ['scipy', 'stats'], \
                "Attempting to assert that base distributions must come from scipy."
        self.base = base
        self.params = []
        self.weights = []

    def __str__(self):
        return 'InfiniteMixture[' + ' '.join(['base: ' + str(self.base),
                                        'params: ' + str(self.params),
                                        'weights: ' + str(self.weights)]) + ']'

    def size(self):
        return len(self.params)

    def log_prob(self, v):
        ret = 0
        for i in range(self.size()):
            part = self.weights[i] * self.base(*self.params[i]).pdf(v)
            ret += part
        return np.log(ret)

    def sample(self, n_samples):
        samples = []
        for _ in range(n_samples):
            idx = np.random.choice(range(self.size()), p=self.weights)
            sample = self.base(*self.params[idx]).rvs(1)
            samples.append(sample)
        return samples

    def sample_n(self, n_samples):
        '''alias for sample method'''
        return self.sample(n_samples)
