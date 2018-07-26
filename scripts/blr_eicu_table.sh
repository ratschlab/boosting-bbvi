#!/bin/bash

echo ==============
echo eICU MORTALITY
echo ==============

ICU_MORTALITY=$HOME/results/bbbvi-leomed/blr/eicu/icu_mortality/seed=*/

echo ROC means \(per iteration\)
xsv cat columns $ICU_MORTALITY/rocs.csv | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").mean(axis=1);
idxs = np.argsort(M);
for i in idxs:
    print(i, M[i])'

echo
echo ROC stds \(per iteration\)
xsv cat columns $ICU_MORTALITY/rocs.csv | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").std(axis=1);
for i,v in enumerate(M):
    print(i, v)'

echo
echo TRAIN LL means \(per iteration\)
xsv cat columns $ICU_MORTALITY/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d, | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").mean(axis=1);
for i,v in enumerate(M):
    print(i, v)'

echo
echo TRAIN LL stds \(per iteration\)
xsv cat columns $ICU_MORTALITY/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d, | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").std(axis=1);
for i,v in enumerate(M):
    print(i, v)'


echo ==============
echo eICU Edward VI
echo ==============

EDWARD_ICU_MORTALITY=$HOME/results/bbbvi-leomed/blr/eicu/icu_mortality/elbo/seed=*/

echo ROC means \(per iteration\)
xsv cat columns $EDWARD_ICU_MORTALITY/rocs.csv | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").mean(axis=0);
print(M)
'

echo ROC stds \(per iteration\)
xsv cat columns $EDWARD_ICU_MORTALITY/rocs.csv | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").std(axis=0);
print(M)
'

echo TRAIN LL means \(per iteration\)
xsv cat columns $EDWARD_ICU_MORTALITY/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d, | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").mean(axis=0)
print(M)
'

echo TRAIN LL stds \(per iteration\)
xsv cat columns $EDWARD_ICU_MORTALITY/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d, | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").std(axis=0)
print(M)
'

echo ==================
echo HOSPITAL MORTALITY
echo ==================

HOSPITAL_MORTALITY=$HOME/results/bbbvi-leomed/blr/eicu/hospital_mortality/seed=*/

echo ROC means \(per iteration\)
xsv cat columns $HOSPITAL_MORTALITY/rocs.csv | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").mean(axis=1);
idxs = np.argsort(M);
for i in idxs:
    print(i, M[i])'

echo
echo ROC stds \(per iteration\)
xsv cat columns $HOSPITAL_MORTALITY/rocs.csv | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").std(axis=1);
for i,v in enumerate(M):
    print(i, v)'

echo
echo TRAIN LL means \(per iteration\)
xsv cat columns $HOSPITAL_MORTALITY/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d, | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").mean(axis=1);
for i,v in enumerate(M):
    print(i, v)'

echo
echo TRAIN LL stds \(per iteration\)
xsv cat columns $HOSPITAL_MORTALITY/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d, | python -c 'import sys; import numpy as np; \
M = np.loadtxt(sys.stdin, delimiter=",").std(axis=1);
for i,v in enumerate(M):
    print(i, v)'
