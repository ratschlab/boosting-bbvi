#!/bin/bash

source activate bbbvi

# Synthetic Data Density
python $BBVI/plot_single_mixture.py \
--qt \
"$HOME/results/bbbvi-leomed/mixture_model_variants/fixed/qt_latest.npz,$HOME/results/bbbvi-leomed/mixture_model_variants/line_search/qt_latest.npz,$HOME/results/bbbvi-leomed/mixture_model_variants/fc/qt_latest.npz,$HOME/results/bbbvi/mixture_model/elbo/qx.npz" \
--target \
~/results/bbbvi-leomed/mixture_model_variants/fc/target_dist.npz \
--outfile bimodal-post.pdf \
--labels "fixed,line search,fully corrective,BBVI" \
--outdir ~/results/bbbvi/paper-figs \
--xlabel '' --ylabel '' --title '' #| xargs open
