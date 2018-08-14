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
flags.DEFINE_boolean('confidence', False, '')
flags.DEFINE_string('times', None, 'path to times for the x-axis')

def plot(xvalues, yvalues, xlabel, ylabel, title, times=False):
    assert(len(xvalues) == len(yvalues))
    out_filename = os.path.join(FLAGS.outdir, FLAGS.outfile)
    fig, ax = plt.subplots()
    ax.plot(xvalues, yvalues, '.-')

    if times:
        xlabel += ' (hours)'
        ax.set_xticks(np.arange(0, max(xvalues), 3600))
        ax.set_xticklabels(range(len(xvalues)))
    else:
        ax.set_xticks(range(len(xvalues)))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.suptitle(title)

    print(out_filename)
    fig.tight_layout()
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
    fig.tight_layout()
    fig.savefig(out_filename)
    plt.close()

def main(argv):
    del argv

    values = np.loadtxt(sys.stdin, delimiter=FLAGS.delimiter)

    # the x-axis is either times (hours) or a range(0, k).
    if FLAGS.times:
        times = []
        with open(FLAGS.times, 'r') as lines:
            for line in lines:
                line = line.strip()
                t = int( float(line) )
                times.append(t)

        t0 = times[0]
        times = [t - t0 for t in times]
        xvalues = times
    else:
        xvalues = range(len(values))

    if len(values.shape) > 1 and values.shape[1] == 2:
        if FLAGS.confidence:
            plot_with_confidence(values, FLAGS.xlabel, FLAGS.ylabel, FLAGS.title)
        else:
            plot(xvalues, values[:,0], FLAGS.xlabel, FLAGS.ylabel, FLAGS.title, FLAGS.times is not None)
    else:
        plot(xvalues, values, FLAGS.xlabel, FLAGS.ylabel, FLAGS.title, FLAGS.times is not None)

if __name__ == "__main__":
    app.run(main)
