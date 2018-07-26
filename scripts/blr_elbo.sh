#!/bin/bash

# chem data
for seed in 8156 18813 25075 7114 31764 15065 10021 8312 24751 14976;
do
    OUTDIR=~/results/bbbvi/blr_elbo/chem/seed=$seed
    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
        --base_dist mvn \
        --exp chem \
        --outdir $OUTDIR \
        --seed $seed" \
        --n_fw_iter 1
done

# wine data
for seed in 2267 30193 23696 6900 18652 32253 3016 31337 17697 8915;
do
    OUTDIR=~/results/bbbvi/blr_elbo/wine/seed=$seed
    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
        --base_dist mvn \
        --exp wine \
        --outdir $OUTDIR \
        --seed $seed" \
        --n_fw_iter 1
done
