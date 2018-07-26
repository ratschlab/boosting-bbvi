#!/usr/bin/python
# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=0

"""Bayesian logistic regression using Hamiltonian Monte Carlo.

We visualize the fit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import os
import sys

from edward.models import Bernoulli, Normal, Empirical, MultivariateNormalDiag
from edward.models import Mixture, Categorical

from scipy.special import expit as sigmoid
from scipy.misc import logsumexp
from sklearn.metrics import roc_auc_score
from mvn import mvn
# TODO add option to switch between mvn & lpl

import utils
import blr_utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("LMO_iter", 600, '')
tf.flags.DEFINE_integer('n_fw_iter', 100, '')
tf.flags.DEFINE_string("outdir", '/tmp', '')
tf.flags.DEFINE_string("fw_variant", 'fixed', '')

tf.flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

def construct_multivariatenormaldiag(dims, iter, name=''):
  loc = tf.get_variable(name + "_loc%d" % iter, initializer=tf.random_normal(dims) + np.random.normal())
  scale = tf.nn.softplus(tf.get_variable(name + "_scale%d" % iter, dims))
  rez = mvn(loc=loc, scale=scale)
  return rez

def main(_):
  ed.set_seed(FLAGS.seed)

  ((Xtrain, ytrain), (Xtest, ytest)) = blr_utils.get_data()
  N,D = Xtrain.shape
  N_test,D_test = Xtest.shape

  weights, q_components = [], []

  g = tf.Graph()
  with g.as_default():
    tf.set_random_seed(FLAGS.seed)
    sess = tf.InteractiveSession()
    with sess.as_default():

      # MODEL
      w = Normal(loc=tf.zeros(D), scale=1.0 * tf.ones(D))

      X = tf.placeholder(tf.float32, [N, D])
      y = Bernoulli(logits=ed.dot(X, w))

      X_test = tf.placeholder(tf.float32, [N_test, D_test]) # TODO why are these test variables necessary?
      y_test = Bernoulli(logits=ed.dot(X_test, w))

      iter = 42 # TODO

      qw = construct_multivariatenormaldiag([D], iter, 'qw')
      inference = ed.KLqp({w: qw}, data={X: Xtrain, y: ytrain})
      tf.global_variables_initializer().run()
      inference.run(n_iter=FLAGS.LMO_iter)

      x_post = ed.copy(y, {w: qw})
      x_post_t = ed.copy(y_test, {w: qw})

      print('log-likelihood train ',ed.evaluate('log_likelihood', data={x_post: ytrain, X:Xtrain}))
      print('log-likelihood test ',ed.evaluate('log_likelihood', data={x_post_t: ytest, X_test:Xtest}))
      print('binary_accuracy train ',ed.evaluate('binary_accuracy', data={x_post: ytrain, X: Xtrain}))
      print('binary_accuracy test ',ed.evaluate('binary_accuracy', data={x_post_t: ytest, X_test: Xtest}))

if __name__ == "__main__":
  tf.app.run()
