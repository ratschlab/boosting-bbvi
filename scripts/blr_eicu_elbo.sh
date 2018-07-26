#!/bin/bash

for seed in 11879 11326 15764 32615 22251 5987 1190 18754 16353 29104;
do
    OUTDIR=~/results/bbbvi/blr/eicu/icu_mortality/elbo/seed=$seed
    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
        --base_dist mvn \
        --exp eicu_icu_mortality \
        --outdir $OUTDIR \
        --seed $seed \
        --n_fw_iter 1 \
        --LMO_iter 10000"
done
