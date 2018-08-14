#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import os
import sys
import numpy as np
import scipy.stats as stats
import plot_utils as utils

from absl import app
from absl import flags

from boosting_bbvi.core.infinite_mixture import InfiniteMixtureScipy

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', '/tmp/', '')
flags.DEFINE_string('outfile', 'mixtures.png', '')
flags.DEFINE_string('title', 'my awesome figure', '')
flags.DEFINE_string('legend_key', '', '')

def construct_mixture_from_params(**kwargs):
    weights = kwargs['weights']
    locs = kwargs['locs']
    diags = kwargs['scale_diags']

    qx = InfiniteMixtureScipy(stats.multivariate_normal)
    qx.weights = weights
    qx.params = list(zip([[l] for l in locs], [[np.dot(d, d)] for d in diags]))

    return qx

class x_log_prob:
    def log_prob(v):
        pi = np.array([0.4, 0.6]).astype(np.float32)
        mus = [[1.], [-1.]]
        stds = [[0.5], [0.5]]

        ret = 0
        K = len(pi)
        for i in range(K):
            this = pi[i] * stats.multivariate_normal(mus[i], np.dot(stds[i], stds[i])).pdf(v)
            ret += this
        return np.log(ret)

def main(argv):
    del argv

    input = [line.strip() for line in sys.stdin]
    qx_params = [np.load(line) for line in input]
    qxs = [x_log_prob]
    qxs.extend([construct_mixture_from_params(**p) for p in qx_params])

    labels = ["target distribution"]
    if FLAGS.legend_key:
        labels.extend([utils.parse_filename(line, FLAGS.legend_key) for line in input])
    else:
        labels.extend(range(len(qxs)))

    fig, ax = plt.subplots()
    grid = np.arange(-2, 2, 0.1)
    for qx,label in zip(qxs, labels):
        probs = [np.exp(qx.log_prob(v)) for v in grid]
        ax.plot(grid, probs, label=label)

    if len(qxs) == 2:
        qx = qxs[1]
        ax.plot([mu[0] for (mu,sig) in qx.params], [0] * len(qx.params), '+')

    fig.suptitle(FLAGS.title)
    plt.legend()
    outpath = os.path.expanduser(os.path.join(FLAGS.outdir, FLAGS.outfile))
    fig.savefig(outpath)
    print(outpath)

if __name__ == "__main__":
    app.run(main)
