#!/bin/bash  

# Hyperparameter Tuning
echo "Hyperparameter Tuning"
python hyperparameter.py

# LOOCV
echo "Leave one out Cross Validation"
python LOOCV.py

# K Fold Cross Validation for k = 5 and 10
echo "K Fold for K = 2,5 and 10"
python kfold.py

# Bootstraps
echo "Bootstrapping"
python bootstrapping.py