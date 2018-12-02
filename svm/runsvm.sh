#!/bin/bash

#Tuning takes a long time.
echo "Tuning:"
python tuning.py
echo ""
#There is an element of randomness so this may not be reproduced.

echo "Bootstrapping:"
python bootstrap.py 5 1000
python bootstrap.py 10 1000
python bootstrap.py 20 1000
echo ""

echo "K-Fold Crass Validation:"
python cv.py 5 1000
python cv.py 10 1000
echo ""

#LOOCV takes an extremely long time.
echo "Leave One Out Cross Validation:"
python loocv.py 1000
echo ""

echo "ROC Curves:"
python cv.py 2 100
python cv.py 2 1000
python cv.py 2 10000
echo ""
