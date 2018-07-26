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
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.color']= 'blue'

import tensorflow as tf
from edward.models import Categorical, MultivariateNormalDiag, Normal, Mixture

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/', '')
flags.DEFINE_string('outfile', 'mixtures.png', '')
flags.DEFINE_string('title', 'my awesome figure', '')

flags.DEFINE_string('ylabel', 'y', '')
flags.DEFINE_string('xlabel', 'x', '')

flags.DEFINE_string('target', None, 'path to target.npz')
flags.mark_flag_as_required('target')

flags.DEFINE_list('qt', [], 'comma-separated list,of,qts to visualize')
flags.DEFINE_list('labels', [], 'labels to be associated with the qts')
flags.DEFINE_list('styles', [], 'styles for each plot')
flags.DEFINE_boolean('widegrid', False, 'range for the x-axis')
flags.DEFINE_boolean('bars', False, 'plot bar chart (loc, weight) for each component')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def construct_mixture_from_params(**kwargs):
    weights = kwargs['weights']
    locs = kwargs['loc']
    diags = kwargs['diags']

    qx = InfiniteMixtureScipy(stats.multivariate_normal)
    qx.weights = weights[0]
    qx.params = list(zip([[l] for l in locs], [[np.dot(d, d)] for d in diags]))

    return qx

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

    if FLAGS.widegrid:
        grid = np.arange(-25, 25, 0.1).astype(np.float32)
    else:
        grid = np.arange(-4, 4, 0.1).astype(np.float32)

    if FLAGS.labels:
        labels = FLAGS.labels
    else:
        labels = ['approximation'] * len(FLAGS.qt)

    if FLAGS.styles:
        styles = FLAGS.styles
    else:
        styles = ['+', 'x', '.', '-']

    grid = np.array([[g] for g in grid]) # package dims for tf
    fig, ax = plt.subplots()
    sess = tf.Session()
    with sess.as_default():
        xprobs = x.log_prob(grid)
        xprobs = tf.exp(xprobs).eval()
        ax.plot(grid, xprobs, label='target', linewidth=2.0)

        if len(FLAGS.qt) == 0:
            eprint("provide some qts to the `--qt` option if you would like to plot them")

        for i,(qt_filename,label) in enumerate(zip(FLAGS.qt, labels)):
            eprint("visualizing %s" % qt_filename)
            qt = deserialize_mixture_from_file(qt_filename)
            qtprobs = tf.exp(qt.log_prob( grid ))
            qtprobs = qtprobs.eval()
            ax.plot(np.squeeze(grid), np.squeeze(qtprobs), styles[i % len(styles)], label=label)

        if len(FLAGS.qt) == 1 and FLAGS.bars:
            locs = [comp.loc.eval() for comp in qt.components]
            ax.plot(locs, [0] * len(locs), '+')

            weights = qt.cat.probs.eval()
            for i in range(len(locs)):
                ax.bar(locs[i], weights[i], .05)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel(FLAGS.xlabel)
    ax.set_ylabel(FLAGS.ylabel)
    fig.suptitle(FLAGS.title)
    legend = plt.legend(loc='upper right', prop={'size': 15}, bbox_to_anchor=(1.08,1))
    outname = os.path.join(os.path.expanduser(FLAGS.outdir), FLAGS.outfile)
    fig.tight_layout()
    fig.savefig(outname, bbox_extra_artists=(legend,), bbox_inches='tight')
    print(outname)

if __name__ == "__main__":
    app.run(main)
