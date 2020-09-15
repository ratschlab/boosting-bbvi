#!/usr/bin/python 

import utils
logger = utils.get_logger()

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf

from edward.models import (
    Categorical, Dirichlet, Empirical, InverseGamma,
    MultivariateNormalDiag, Normal, ParamMixture, Mixture)

import edward as ed
import copy
import scipy.stats as stats
from scipy.misc import logsumexp as logsumexp

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp', 'directory to store all the results, models, plots, etc.')
flags.DEFINE_integer('seed', 0, 'The random seed to use for everything.')
flags.DEFINE_string('exp', 'mixture', 'select from [mixture, s_and_s (aka spike and slab), many]')
flags.DEFINE_integer('n_iter', 10000, 'number of iterations for VI')

ed.set_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

# what we are trying to fit
if FLAGS.exp == 'mixture':
    pi = np.array([[0.4, 0.6]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.6], [.6]]
elif FLAGS.exp == 's_and_s':
    pi = np.array([[0.4, 0.6]]).astype(np.float32)
    mus = [[1.], [-1.]]
    stds = [[.1], [10.]]
elif FLAGS.exp == 'many':
    mus = np.array([[5.0], [10.0], [20.0], [-2]]).astype(np.float32)
    stds = np.array([[2], [2], [1], [1]]).astype(np.float32)
    pi = np.array([[1.0/3, 1.0/4, 1.0/4, 1.0/6]]).astype(np.float32)
else:
    raise KeyError("undefined experiment")

# global settings
N = 500

def build_toy_dataset(N, D=1):
    x = np.zeros((N, D), dtype=np.float32)
    ks = np.zeros(N, dtype=np.int)
    for n in range(N):
        k = np.argmax(np.random.multinomial(1, pi[0]))
        x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
        ks[n] = k
    return x,ks

def construct_normal(dims, iter, name='', sample_shape=N):
    loc = tf.get_variable(name + "_loc%d" % iter, initializer=tf.random_normal(dims) + \
            np.random.normal())
    scale = tf.get_variable(name + "_scale%d" % iter, initializer=tf.random_normal(dims))
    return Normal(loc=loc, scale=tf.nn.softplus(scale), sample_shape=N)

def setup_outdir():
    outdir = FLAGS.outdir
    outdir = os.path.expanduser(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir

if __name__ == "__main__":
    x_train, components = build_toy_dataset(N)
    n_examples, n_features = x_train.shape

    # build model
    xcomps = [Normal(loc=tf.convert_to_tensor(mus[i]),
                    scale=tf.convert_to_tensor(stds[i])) for i in range(len(mus))]
    x = Mixture(cat=Categorical(probs=tf.convert_to_tensor(pi)), components=xcomps, sample_shape=N)

    qx = construct_normal([n_features], 42, 'qx')

    inference = ed.KLqp({x: qx})
    inference.run(n_iter=FLAGS.n_iter)

    # save the target
    outdir = setup_outdir()
    np.savez(os.path.join(outdir, 'target_dist.npz'),
            pi=pi, mus=mus, stds=stds)

    # save the approximation
    outdir = setup_outdir()
    for_serialization = {'locs': np.array([qx.mean().eval()]),
                         'scale_diags': np.array([qx.stddev().eval()])}
    qt_outfile = os.path.join(outdir, 'qx.npz')
    np.savez(qt_outfile, weights=[1.0], **for_serialization)
    logger.info("saving qx to, %s" % qt_outfile)
