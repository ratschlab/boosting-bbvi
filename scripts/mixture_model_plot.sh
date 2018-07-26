#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: path/to/directory/of/results"
    exit -1
fi

INPUT=$1
SRC=src

ls $INPUT/lambda=1.0,fw_variant=*,decay=log/losses.csv | xargs -L1 echo | python $SRC/plot_losses.py --title "losses" --xlabel 'Frank Wolfe Iteration' --ylabel 'KL(q,p_x)' --legend_key fw_variant | xargs open

ls $INPUT/lambda=1.0,fw_variant=*,decay=log/qx_params_latest.npz | xargs -L1 echo | python $SRC/plot_mixtures.py --title "Distributions" --legend_key fw_variant | tee | xargs open
