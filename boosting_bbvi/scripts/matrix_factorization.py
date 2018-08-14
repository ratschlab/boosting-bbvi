#!/usr/bin/python

"""Probabilistic matrix factorization using variational inference.

Visualizes the actual and the estimated rating matrices as heatmaps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib
matplotlib.use('Agg')

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as sio

from edward.models import Normal, MultivariateNormalDiag, Mixture, Categorical, ParamMixture

from boosting_bbvi.core.utils import block_diagonal
import boosting_bbvi.core.relbo as relbo
from boosting_bbvi.core.mvn import mvn

flags = tf.app.flags
FLAGS = tf.flags.FLAGS
flags.DEFINE_integer("N", 50, "Number of users.")
flags.DEFINE_integer("M", 60, "Number of movies.")
flags.DEFINE_integer("D", 3, "Number of latent factors.")
flags.DEFINE_integer('n_fw_iter', 1000, '')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_string('outdir', '/tmp', '')
flags.DEFINE_float('mask_ratio', 0.5, '')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

def build_toy_dataset(U, V, N, M, noise_std=0.1):
  R = np.dot(np.transpose(U), V) + np.random.normal(0, noise_std, size=(N, M))
  return R

def get_indicators(N, M, prob_std=0.5):
  ind = np.random.binomial(1, prob_std, (N, M))
  return ind

def flatten(a):
    n,m = a.shape
    return np.reshape(a, n*m)

def build_mixture(weights, qus):
  cat = Categorical(probs=tf.convert_to_tensor(weights))
  comps = [mvn(loc=tf.convert_to_tensor(c['loc']),scale=tf.convert_to_tensor(c['scale'])) for c in qus]

  mix = Mixture(cat=cat, components=comps)
  return mix

def get_fw_iterates(iter, weights, target, qUVt_components):
  ret = {}
  if iter > 0:
    ret = {target: build_mixture(weights, qUVt_components)}
  return ret

def update_iterate(components, s):
  loc = s.loc.eval()
  scale = s.scale.eval()
  components.append({'loc': loc, 'scale': scale})
  return components

def main(_):
  # true latent factors
  U_true = np.random.randn(FLAGS.D, FLAGS.N)
  V_true = np.random.randn(FLAGS.D, FLAGS.M)

  ## DATA
  #R_true = build_toy_dataset(U_true, V_true, FLAGS.N, FLAGS.M)
  #I_train = get_indicators(FLAGS.N, FLAGS.M)
  #I_test = 1 - I_train
  #N = FLAGS.N
  #M = FLAGS.M

  #tr = sio.loadmat(os.path.expanduser("~/data/bbbvi/trainData1.mat"))['X']
  #te = sio.loadmat(os.path.expanduser("~/data/bbbvi/testData1.mat"))['X']
  #tr = tr[:,:100]
  #te = te[:,:100]
  #I_train = tr != 0
  #I_test = te != 0
  #R_true = (tr + te).astype(np.float32)
  #N,M = R_true.shape

  tr = sio.loadmat(os.path.expanduser("~/data/bbbvi/cbcl.mat"))['V']
  te = sio.loadmat(os.path.expanduser("~/data/bbbvi/cbcl.mat"))['V']
  #I_train = np.ones(tr.shape)
  #I_test = np.ones(tr.shape)
  R_true = tr
  N,M = tr.shape
  D = FLAGS.D
  I_train = get_indicators(N, M, FLAGS.mask_ratio)
  I_test = 1 - I_train

  it_best = 0
  weights, qUVt_components, mses = [], [], []
  test_mses, test_lls = [], []
  for iter in range(FLAGS.n_fw_iter):
    print("iter", iter)
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(FLAGS.seed)
        sess = tf.InteractiveSession()
        with sess.as_default():
          # MODEL
          I = tf.placeholder(tf.float32, [N, M])

          scale_uv = tf.concat([tf.ones([FLAGS.D, N]), tf.ones([FLAGS.D, M])], axis=1)
          mean_uv = tf.concat([tf.zeros([FLAGS.D, N]), tf.zeros([FLAGS.D, M])], axis=1)

          UV = Normal(loc=mean_uv, scale=scale_uv)
          R = Normal(loc=tf.matmul(tf.transpose(UV[:, :N]), UV[:, N:]) * I, scale=tf.ones([N, M]))
          mean_quv = tf.concat([tf.get_variable("qU/loc", [FLAGS.D, N]), tf.get_variable("qV/loc", [FLAGS.D, M])],
                               axis=1)
          scale_quv = tf.concat([tf.nn.softplus(tf.get_variable("qU/scale", [FLAGS.D, N])),
                                 tf.nn.softplus(tf.get_variable("qV/scale", [FLAGS.D, M]))], axis=1)

          qUV = Normal(loc=mean_quv, scale=scale_quv)

          inference = relbo.KLqp({UV: qUV}, data={R: R_true, I: I_train},
                        fw_iterates=get_fw_iterates(iter, weights, UV, qUVt_components),
                        fw_iter=iter)
          inference.run(n_iter=100)

          gamma = 2. / (iter + 2.)
          weights = [(1. - gamma) * w for w in weights]
          weights.append(gamma)

          qUVt_components = update_iterate(qUVt_components, qUV)

          qUV_new = build_mixture(weights, qUVt_components)
          qR = Normal(loc=tf.matmul(tf.transpose(qUV_new[:, :N]), qUV_new[:, N:]), scale=tf.ones([N, M]))

          # CRITICISM
          test_mse = ed.evaluate('mean_squared_error', data={qR: R_true, I: I_test.astype(bool)})
          test_mses.append(test_mse)
          print('test mse', test_mse)

          test_ll = ed.evaluate('log_lik', data={qR: R_true.astype('float32'), I: I_test.astype(bool)})
          test_lls.append(test_ll)
          print('test_ll', test_ll)

          np.savetxt(os.path.join(FLAGS.outdir, 'test_mse.csv'), test_mses, delimiter=',')
          np.savetxt(os.path.join(FLAGS.outdir, 'test_ll.csv'), test_lls, delimiter=',')

if __name__ == "__main__":
  tf.app.run(main)
