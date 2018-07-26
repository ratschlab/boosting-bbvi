#!/usr/bin/python

import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == "__main__":
    npz_filename, outname, title, xlabel, ylabel = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    losses = np.load(npz_filename)
    losses = losses['logliks']

    fig, ax = plt.subplots()
    ax.plot(range(len(losses)), losses, '.-')
    ax.set_xticks(range(len(losses)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    fig.savefig(outname)
