#!/bin/bash

echo "first iteration mean std, last iteration mean std TEST MSE"

echo D = 3
for f in `ls ~/results/bbbvi-leomed/matrix_factorizaton/D=3/seed=*/test_mse.csv`;
do
    echo $(head -n1 $f), $(tail -n1 $f)
done | python -c "import numpy as np; import sys; foo = np.array([ (float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in sys.stdin]); print(list(zip(np.mean(foo, axis=0), np.std(foo, axis=0))))"

echo D = 5
for f in `ls ~/results/bbbvi-leomed/matrix_factorizaton/D=5/seed=*/test_mse.csv`;
do
    echo $(head -n1 $f), $(tail -n1 $f)
done | python -c "import numpy as np; import sys; foo = np.array([ (float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in sys.stdin]); print(list(zip(np.mean(foo, axis=0), np.std(foo, axis=0))))"

echo D = 10
for f in `ls ~/results/bbbvi-leomed/matrix_factorizaton/D=10/seed=*/test_mse.csv`;
do
    echo $(head -n1 $f), $(tail -n1 $f)
done | python -c "import numpy as np; import sys; foo = np.array([ (float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in sys.stdin]); print(list(zip(np.mean(foo, axis=0), np.std(foo, axis=0))))"

echo "first iteration mean std, last iteration mean std TEST LL"

echo D = 3
for f in `ls ~/results/bbbvi-leomed/matrix_factorizaton/D=3/seed=*/test_ll.csv`;
do
    echo $(head -n1 $f), $(tail -n1 $f)
done | python -c "import numpy as np; import sys; foo = np.array([ (float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in sys.stdin]); print(list(zip(np.mean(foo, axis=0), np.std(foo, axis=0))))"

echo D = 5
for f in `ls ~/results/bbbvi-leomed/matrix_factorizaton/D=5/seed=*/test_ll.csv`;
do
    echo $(head -n1 $f), $(tail -n1 $f)
done | python -c "import numpy as np; import sys; foo = np.array([ (float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in sys.stdin]); print(list(zip(np.mean(foo, axis=0), np.std(foo, axis=0))))"

echo D = 10
for f in `ls ~/results/bbbvi-leomed/matrix_factorizaton/D=10/seed=*/test_ll.csv`;
do
    echo $(head -n1 $f), $(tail -n1 $f)
done | python -c "import numpy as np; import sys; foo = np.array([ (float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in sys.stdin]); print(list(zip(np.mean(foo, axis=0), np.std(foo, axis=0))))"
