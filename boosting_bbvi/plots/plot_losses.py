#!/usr/bin/python

import os
import sys

import numpy as np
import plot_utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/', '')
flags.DEFINE_string('outfile', 'losses.png', '')
flags.DEFINE_string('title', 'my awesome figure', '')
flags.DEFINE_string('legend_key', '', '')

flags.DEFINE_string('xlabel', 'x', '')
flags.DEFINE_string('ylabel', 'y', '')

def main(argv):
    del argv

    filenames = [line.strip() for line in sys.stdin]
    fig, ax = plt.subplots()
    ax.set_xlabel(FLAGS.xlabel)
    ax.set_ylabel(FLAGS.ylabel)
    fig.suptitle(FLAGS.title)

    if FLAGS.legend_key:
        labels = [utils.parse_filename(fn, FLAGS.legend_key) for fn in filenames]
    else:
        labels = range(len(filenames))

    for filename,label in zip(filenames, labels):
        values = np.loadtxt(filename, delimiter=',')
        means, stds = values[:,0], values[:,1]
        line, = ax.plot(range(len(values)), means, '.-', label=label)
        ax.fill_between(range(len(values)), means + stds, means - stds, color=line.get_color(), alpha=0.5)

    plt.legend()

    out_filename = os.path.join(FLAGS.outdir, FLAGS.outfile)
    print(out_filename)
    fig.savefig(out_filename)
    plt.close()

if __name__ == "__main__":
    app.run(main)
