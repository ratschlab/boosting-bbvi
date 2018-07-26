#!/bin/bash

UUID=$(date +%s)

for reg in `seq 0.05 0.10 1.00`; do
    for variant in 'fixed' 'line_search';
    do
        for exp in 'mixture' 's_and_s' 'many' 'balanced';
        do
            for anneal in 'linear' 'constant' 'log';
            do
                OUTDIR=~/results/bbbvi/large/mixture_model_relbo/$UUID/relbo_reg=${reg}_relboanneal=${anneal}_variant=${variant}_exp=${exp}_anneal=${anneal}/
                mkdir -p $OUTDIR
                echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/mixture_model_relbo.py --relbo_reg $reg --relbo_anneal $anneal --exp $exp --fw_variant $variant --outdir ${OUTDIR}"
            done
        done
    done

done
