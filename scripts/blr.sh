#!/bin/bash

 chem data
for seed in 25001 23689 14495 592 21134 4712 8271 16678 10628 21730;
do
    OUTDIR=~/results/bbbvi/blr/chem/seed=$seed
    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
        --base_dist lpl \
        --exp chem \
        --outdir $OUTDIR \
        --seed $seed"
done

# wine data
for seed in 17000 30876 23719 5662 16433 30769 29038 1323 23213 20176;
do
    OUTDIR=~/results/bbbvi/blr/wine/seed=$seed
    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
        --base_dist lpl \
        --exp wine \
        --outdir $OUTDIR \
        --seed $seed"
done

