# General

This is the source code for "Boosting Black Box Variational Inference," by
Francesco Locatello, Gideon Dresdner, Rajiv Khanna, Isabel Valera, Gunnar
Rätsch. https://arxiv.org/abs/1806.02185.
```
@article{locatello2018boosting,
  title={Boosting Black Box Variational Inference},
  author={Locatello, Francesco and Dresdner, Gideon and Khanna, Rajiv and Valera, Isabel and R{\"a}tsch, Gunnar},
  journal={arXiv preprint arXiv:1806.02185},
  year={2018}
}
```

# Setup

1. Setup the dependencies using conda:
```
conda env create -n bbbvi --file conda-env.txt
```

2. Activate the environment
```
source activate bbbvi
```

3. Install the package for development
```
python setup.py develop
```

# Run

## Bayesian Logistic Regression

To recreate the Bayesian Linear Regression results in Table 1 of the paper, run
```
python blr.py \
    --base_dist lpl \
    --exp $EXPERIMENT \
    --outdir $OUTDIR \
    --seed $seed"
```
Where `$EXPERIMENT` is either `chem` or `wine`.

## Bayesian Matrix Factorization

To recreate the Bayesian matrix factorization results in Table 1 of the paper, run
```
python matrix_factorization.py --D $D --outdir $OUTDIR --seed $seed
```

## Toy data (mixture model)

To recreate Figure 1 of the experiment run
```
python mixture_model_relbo.py \
    --relbo_reg 1.0 \
    --relbo_anneal linear \
    --exp mixture \
    --fw_variant $variant \
    --outdir $OUTDIR"
```
Where `$variant` is `fixed`, `line_search`, or `fc`.

# License

This project is licensed under the MIT License - see the LICENSE file for details.
