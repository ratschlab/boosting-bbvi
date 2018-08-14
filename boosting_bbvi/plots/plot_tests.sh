#!/bin/bash

SRC=../src

ls test_mixture_model_results/1/fw_variant=*/test_losses.csv | xargs -L1 echo | python $SRC/plot_losses.py --title "Good Night Moon" --xlabel 'xgen' --ylabel 'hippo' | xargs open

ls test_mixture_model_results/1/fw_variant=*/qx_params_latest.npz | xargs -L1 echo | python $SRC/plot_mixtures.py --title "garb" --legend_key fw_variant | xargs open
