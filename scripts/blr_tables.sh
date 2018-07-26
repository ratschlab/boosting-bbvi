#!/bin/bash

BASEDIR=~/results/bbbvi-leomed/blr

echo CHEM TEST LOG LIKELIHOODS
echo iter mean std
xsv cat columns $BASEDIR/chem/seed=*/test_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d',' | python -c 'import sys; import numpy as np; M = np.loadtxt(sys.stdin, delimiter=","); ret = list(zip(np.mean(M, axis=1), np.std(M, axis=1))); [print(l[0], l[1]) for l in ret]' | awk '{print NR " " $0}' | sort -k2n | tr ' ' '\t'

echo
echo CHEM TRAIN LOG LIKELIHOODS
echo iter mean std
xsv cat columns $BASEDIR/chem/seed=*/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d',' | python -c 'import sys; import numpy as np; M = np.loadtxt(sys.stdin, delimiter=","); ret = list(zip(np.mean(M, axis=1), np.std(M, axis=1))); [print(l[0], l[1]) for l in ret]' | awk '{print NR " " $0}' | sort -k2n

echo
echo CHEM TEST AUROC
echo iter mean std
xsv cat columns $BASEDIR/chem/seed=*/rocs.csv | python -c 'import sys; import numpy as np; M = np.loadtxt(sys.stdin, delimiter=","); ret = list(zip(np.mean(M, axis=1), np.std(M, axis=1))); [print(l[0], l[1]) for l in ret]' | awk '{print NR " " $0}' | sort -k2n

echo WINE TEST LOG LIKELIHOODS
echo iter mean std
xsv cat columns $BASEDIR/wine/seed=*/test_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d',' | python -c 'import sys; import numpy as np; M = np.loadtxt(sys.stdin, delimiter=","); ret = list(zip(np.mean(M, axis=1), np.std(M, axis=1))); [print(l[0], l[1]) for l in ret]' | awk '{print NR " " $0}' | sort -k2n | tr ' ' '\t'

echo
echo WINE TRAIN LOG LIKELIHOODS
echo iter mean std
xsv cat columns $BASEDIR/wine/seed=*/train_lls.csv | cut -f1,3,5,7,9,11,13,15,17,19 -d',' | python -c 'import sys; import numpy as np; M = np.loadtxt(sys.stdin, delimiter=","); ret = list(zip(np.mean(M, axis=1), np.std(M, axis=1))); [print(l[0], l[1]) for l in ret]' | awk '{print NR " " $0}' | sort -k2n

echo
echo WINE TEST AUROC
echo iter mean std
xsv cat columns $BASEDIR/wine/seed=*/rocs.csv | python -c 'import sys; import numpy as np; M = np.loadtxt(sys.stdin, delimiter=","); ret = list(zip(np.mean(M, axis=1), np.std(M, axis=1))); [print(l[0], l[1]) for l in ret]' | awk '{print NR " " $0}' | sort -k2n

echo
echo WINE duality gaps
echo iter mean std
xsv cat columns $BASEDIR/wine/seed=*/gaps.csv | python -c 'import sys; import numpy as np; M = np.loadtxt(sys.stdin, delimiter=","); ret = list(zip(np.mean(M, axis=1), np.std(M, axis=1))); [print(l[0], l[1]) for l in ret]' | awk '{print NR " " $0}' | sort -k2n
