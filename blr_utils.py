#!/usr/bin/python

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
import relbo

from scipy.special import expit as sigmoid
from scipy.misc import logsumexp
from sklearn.metrics import roc_auc_score
from mvn import mvn

import utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('exp', 'chem', '')

def build_1d(N=40, noise_std=0.1):
  D = 1
  X = np.linspace(-6, 6, num=N)
  y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  return X, y

def build_1d_bimodal(N=40, noise_std=0.1):
  D = 1
  X = np.linspace(-6, 60, num=N)
  y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  X = (X - 4.0) / 4.0

  X = np.append(X, X[y == 0] + 20)
  y = np.append(y, np.zeros((y[y == 0]).shape))

  X = X.reshape((X.shape[0], D))

  plt.scatter(X,y)
  plt.show()

  return X, y

def build_linearly_separable():
  Os = np.array([ np.array([3.0, 3.0]) + np.random.normal([0, 0], [2., 2.]) for _ in range(100) ])
  Xs = np.array([ np.array([-3.0, -3.0]) + np.random.normal([0, 0], [2., 2.]) for _ in range(100) ])

  X = np.vstack((Os, Xs))
  y = np.concatenate([np.zeros(100), np.ones(100)])
  idxs = np.arange(200)
  np.random.shuffle(idxs)

  return X[idxs,:], y[idxs]

def build_xs_and_os():
  Os1 = np.array([ np.array([3.0, 3.0]) + np.random.normal([0, 0], [1., 1.]) for _ in range(100) ])
  Xs1 = np.array([ np.array([3.0, -3.0]) + np.random.normal([0, 0], [1., 1.]) for _ in range(100) ])

  Os2 = np.array([np.array([-3.0, -3.0]) + np.random.normal([0, 0], [1., 1.]) for _ in range(100)])
  Xs2 = np.array([np.array([-3.0, 3.0]) + np.random.normal([0, 0], [1., 1.]) for _ in range(100)])

  Os = np.vstack((Os1, Os2))
  Xs = np.vstack((Xs1, Xs2))

  X = np.vstack((Os, Xs))
  y = np.concatenate([np.zeros(200), np.ones(200)])
  idxs = np.arange(400)
  np.random.shuffle(idxs)

  return X[idxs,:], y[idxs]

def load_chem_data(path):
  dat = np.load(path)
  if 'X' in dat:
      X = dat['X']
  else:
      X = sp.csr_matrix((dat['data'], dat['indices'], dat['indptr']),
                        shape=dat['shape'])
  y = dat['y']
  y[y <0] = 0
  return X, y

def get_chem_data():
  traindatapath = os.path.expanduser('ds1.100_train.npz')
  Xtrain, ytrain = load_chem_data(traindatapath)
  testdatapath = os.path.expanduser('ds1.100_test.npz')
  Xtest, ytest = load_chem_data(testdatapath)
  return ((Xtrain, ytrain), (Xtest, ytest))

def construct_multivariatenormaldiag(dims, iter, name=''):
  loc = tf.get_variable(name + "_loc%d" % iter, initializer=tf.random_normal(dims) + np.random.normal())
  scale = tf.nn.softplus(tf.get_variable(name + "_scale%d" % iter, dims))
  rez = mvn(loc=loc, scale=scale)
  return rez

def setup_outdir():
  outdir = FLAGS.outdir
  outdir = os.path.expanduser(outdir)
  os.makedirs(outdir, exist_ok=True)
  return outdir

def build_mixture(weights,components):
  cat = Categorical(probs=tf.convert_to_tensor(weights))
  comps = [mvn(loc=tf.convert_to_tensor(c['loc']),
                 scale=tf.convert_to_tensor(c['scale'])) for c in components]
  mix = Mixture(cat=cat, components=comps)
  return mix

def add_bias_column(X):
  N,D = X.shape
  ret = np.c_[X, np.ones(N)]
  return ret

def load_wine_data():
  basepath = os.path.expanduser("~/data/bbbvi/wine")
  filename = os.path.join(basepath, 'train_test_split.npz')
  data = np.load(filename)
  return (data['Xtrain'], data['ytrain']), (data['Xtest'], data['ytest'])

def load_eicu(task='icu_mortality'):
  basepath = "/cluster/work/grlab/projects/bbbvi/eicu/train_test_split.npz"
  data = np.load(basepath)

  # N.B. we only deal with classification, no regression on length of icu stay.
  tasks = ['icu_mortality', 'hospital_mortality', 'length_of_icu_stay']
  col = tasks.index(task)

  return (data['xtrain'], data['ytrain'][:,col]), (data['xtest'], data['ytest'][:,col])

def get_data():
  if FLAGS.exp == 'synthetic_linearly_sep':
    Xtrain,ytrain = build_linearly_separable()
    Xtest,ytest = build_linearly_separable()

    #fig, ax = plt.subplots() # TODO dup below
    #ax.scatter(Xtrain[ytrain == 0,0], Xtrain[ytrain == 0,1], marker='o')
    #ax.scatter(Xtrain[ytrain == 1,0], Xtrain[ytrain == 1,1], marker='x')
    #plt.show()
  elif FLAGS.exp == 'synthetic_not_linearly_sep':
    Xtrain,ytrain = build_xs_and_os()
    Xtest,ytest = build_xs_and_os()

    #fig, ax = plt.subplots()
    #ax.scatter(Xtrain[ytrain == 0,0], Xtrain[ytrain == 0,1], marker='o')
    #ax.scatter(Xtrain[ytrain == 1,0], Xtrain[ytrain == 1,1], marker='x')
    #plt.show()
  elif FLAGS.exp == 'synthetic_1d':
    return build_1d(), build_1d()
  elif FLAGS.exp == 'synthetic_1d_bimodal':
    return build_1d_bimodal(), build_1d_bimodal()
  elif FLAGS.exp == 'chem':
    ((Xtrain, ytrain), (Xtest, ytest)) = get_chem_data()
  elif FLAGS.exp == 'wine':
    ((Xtrain, ytrain), (Xtest, ytest)) = load_wine_data()
  elif FLAGS.exp == 'eicu_icu_mortality':
    ((Xtrain, ytrain), (Xtest, ytest)) = load_eicu('icu_mortality')
  elif FLAGS.exp == 'eicu_hospital_mortality':
    ((Xtrain, ytrain), (Xtest, ytest)) = load_eicu('hospital_mortality')
  else:
    raise Exception("unknown experiment")
  return (Xtrain, ytrain), (Xtest, ytest)
