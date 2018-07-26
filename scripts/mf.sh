#!/bin/bash

source activate bbbvi

for D in 3 5 10;
do
    for seed in 30569 15479 21801 6842 14712 1504 14052 28288 12898 30307;
    do
        OUTDIR=~/results/bbbvi/matrix_factorizaton/D=$D/seed=$seed
        mkdir -p $OUTDIR
        echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/matrix_factorization_francesco.py --D $D --outdir $OUTDIR --seed $seed"
    done
done

