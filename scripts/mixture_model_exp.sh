#!/bin/bash

CONDA="source activate bbbvi"
OUTDIR="~/results/bbbvi/mixture_model"
TIMESTAMP=$(date +%s)

for lambda in '0.01' '0.1' '1.0' '10.'; do
    for fw_variant in 'fixed' 'line_search_dkl' 'line_search_binary'; do
        for decay in 'linear' 'log' 'squared'; do
            DIR="${TIMESTAMP}/lambda=${lambda},fw_variant=${fw_variant},decay=${decay}"
            $CONDA; python src/mixture_model_prototype.py --Lambda $lambda --fw_variant fixed --decay log --outdir ${OUTDIR}/${DIR}/ &
        done
    done
done
