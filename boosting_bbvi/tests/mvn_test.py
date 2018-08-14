#!/usr/bin/python

from edward.models import MultivariateNormalDiag
import tensorflow as tf
import numpy as np

import boosting_bbvi.core.mvn as mvn

def test_mvn_same_as_edward_mvn():
    loc = np.zeros(5)
    scale = np.ones(5)

    A = mvn.mvn(loc=loc, scale=scale)
    B = MultivariateNormalDiag(loc=loc, scale_diag=scale)

    M = np.random.rand(5,5)
    tf.InteractiveSession()

    assert( tf.reduce_sum(A.log_prob(M)).eval() - tf.reduce_sum(B.log_prob(M)).eval() < 1e-6)

if __name__ == "__main__":
    test_mvn_same_as_edward_mvn()
