#!/usr/bin/python

from __future__ import print_function

import os
import sys
import numpy as np
from infinite_mixture import InfiniteMixtureScipy
import scipy.stats as stats
import plot_utils as utils

from absl import app
from absl import flags

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf
from edward.models import Categorical, MultivariateNormalDiag, Normal, Mixture

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/', '')
flags.DEFINE_string('outfile', 'residual.png', '')
flags.DEFINE_string('title', 'my awesome figure', '')

flags.DEFINE_string('target', None, 'path to target.npz')
flags.mark_flag_as_required('target')

flags.DEFINE_string('qt', None, 'qt to use in computing the residual')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def deserialize_mixture_from_file(filename):
    qt_deserialized = np.load(filename)
    locs = qt_deserialized['locs'].astype(np.float32)
    scale_diags = qt_deserialized['scale_diags'].astype(np.float32)
    weights = qt_deserialized['weights'].astype(np.float32)

    #q_comps = [Normal(loc=loc[0], scale=tf.nn.softmax(scale_diag)[0]) \
    q_comps = [Normal(loc=loc[0], scale=scale_diag[0]) \
                for loc, scale_diag in zip(locs, scale_diags)]
    cat = Categorical(probs=tf.convert_to_tensor(weights))
    q_latest = Mixture(cat=cat, components=q_comps)
    return q_latest

def deserialize_target_from_file(filename):
    qt_deserialized = np.load(filename)
    mus = qt_deserialized['mus'].astype(np.float32)
    stds = qt_deserialized['stds'].astype(np.float32)
    pi = qt_deserialized['pi'].astype(np.float32)

    cat = Categorical(probs=tf.convert_to_tensor(pi))
    target_comps = [Normal(loc=tf.convert_to_tensor(mus[i]),
                    scale=tf.convert_to_tensor(stds[i])) for i in range(len(mus))]
    return Mixture(cat=cat, components=target_comps)

def main(argv):
    del argv

    x = deserialize_target_from_file(FLAGS.target)

    grid = np.arange(-2, 2, 0.1).astype(np.float32)
    grid = np.array([[g] for g in grid]) # package dims for tf
    fig, ax = plt.subplots()
    sess = tf.Session()
    with sess.as_default():
        xprobs = x.log_prob(grid)
        xprobs = np.squeeze(xprobs.eval())
        #ax.plot(grid, xprobs, label='target')

        qt = deserialize_mixture_from_file(FLAGS.qt)
        qtprobs = qt.log_prob( np.array([[g] for g in grid]) )
        qtprobs = np.squeeze(qtprobs.eval())
        residual = np.exp(xprobs - qtprobs)
        ax.plot(residual, label='residual')

    fig.suptitle(FLAGS.title)
    plt.legend()
    outname = os.path.join(os.path.expanduser(FLAGS.outdir), FLAGS.outfile)
    fig.savefig(outname)
    print(outname)

if __name__ == "__main__":
    app.run(main)

