#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import os
import sys
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/', '')
flags.DEFINE_string('outfile', 'values.png', '')
flags.DEFINE_string('xlabel', 'x', '')
flags.DEFINE_string('ylabel', 'y', '')
flags.DEFINE_string('title', 'my awesome figure', '')
flags.DEFINE_string('delimiter', ',', '')
flags.DEFINE_list('labels', [], '')
flags.DEFINE_boolean('confidence', False, '')
flags.DEFINE_string('times', None, 'path to csv with timestamps (order needs to match)')

def plot(values, xlabel, ylabel, title, xvalues=None):
    n,d = values.shape

    if len(FLAGS.labels) == 0:
        labels = range(d)
    else:
        labels = FLAGS.labels

    fig, ax = plt.subplots()
    for i in range(d):
        vs = values[:,i]
        ax.plot(range(len(vs)), vs, '.-', label=labels[i])

    if not xvalues:
        xvalues = range(len(values))

    ax.set_xticks(xvalues)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.legend()

    out_filename = os.path.join(FLAGS.outdir, FLAGS.outfile)
    print(out_filename)
    fig.savefig(out_filename)
    plt.close()

def plot_with_confidence(values, xlabel, ylabel, title):
    '''
    One of those plots with a mean line and a fill between `mean + std` and
    `mean - std`.
    '''
    out_filename = os.path.join(FLAGS.outdir, FLAGS.outfile)
    fig, ax = plt.subplots()

    means, stds = values[:,0], values[:,1]
    line, = ax.plot(range(len(values)), means, '.-')
    ax.fill_between(range(len(values)), means + stds, means - stds, color=line.get_color(), alpha=0.5)

    fig.suptitle(title)
    ax.set_xticks(range(len(values)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    print(out_filename)
    fig.savefig(out_filename)
    plt.close()

def main(argv):
    del argv

    values = np.loadtxt(sys.stdin, delimiter=FLAGS.delimiter)
    if len(values.shape) == 1:
        values = np.reshape(values, (values.shape[0], 1))

    if FLAGS.confidence:
        plot_with_confidence(values, FLAGS.xlabel, FLAGS.ylabel, FLAGS.title)
    else:
        plot(values, FLAGS.xlabel, FLAGS.ylabel, FLAGS.title)

if __name__ == "__main__":
    app.run(main)

