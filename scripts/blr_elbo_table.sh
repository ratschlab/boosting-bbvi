#!/bin/bash

echo ROC
xsv cat columns ~/results/bbbvi-leomed/blr_elbo/chem/seed=*/rocs.csv | python -c "import sys; import numpy as np; print(np.loadtxt(sys.stdin, delimiter=',').mean())"

xsv cat columns ~/results/bbbvi-leomed/blr_elbo/chem/seed=*/rocs.csv | python -c "import sys; import numpy as np; print(np.loadtxt(sys.stdin, delimiter=',').std())"

echo TRAIN LL
cat ~/results/bbbvi-leomed/blr_elbo/chem/seed=*/train_lls.csv | cut -f1 -d, | xsv cat columns | python -c "import sys; import numpy as np; print(np.loadtxt(sys.stdin, delimiter=',').mean())"

cat ~/results/bbbvi-leomed/blr_elbo/chem/seed=*/train_lls.csv | cut -f1 -d, | xsv cat columns | python -c "import sys; import numpy as np; print(np.loadtxt(sys.stdin, delimiter=',').std())"
