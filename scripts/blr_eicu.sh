#!/bin/bash

# LPL
#for seed in 22710 21910 8622 21519 23223 24567 22340 20851 24803 4519;
#do
#    OUTDIR=~/results/bbbvi/blr/eicu/icu_mortality/seed=$seed
#    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
#        --base_dist lpl \
#        --exp eicu_icu_mortality \
#        --outdir $OUTDIR \
#        --seed $seed"
#done
#
#for seed in 8565 11273 7665 23281 15901 384 9027 11843 21089 29121;
#do
#    OUTDIR=~/results/bbbvi/blr/eicu/hospital_mortality/seed=$seed
#    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
#        --base_dist lpl \
#        --exp eicu_hospital_mortality \
#        --outdir $OUTDIR \
#        --seed $seed"
#done

# Regression, not classification =)
#OUTDIR=~/results/bbbvi/blr/eicu/eicu_length_of_stay
#echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
#    --base_dist lpl \
#    --exp eicu_length_of_stay \
#    --outdir $OUTDIR"

# MVN
for seed in 22710 21910 8622 21519 23223 24567 22340 20851 24803 4519;
do
    OUTDIR=~/results/bbbvi/blr/eicu/icu_mortality/mvn/seed=$seed
    echo "source activate bbbvi; python /cluster/home/dresdnerg/projects/projects2017-Boosting-VI/Code/bbbvi/src/blr.py \
        --base_dist mvn \
        --exp eicu_icu_mortality \
        --outdir $OUTDIR \
        --seed $seed"
done
