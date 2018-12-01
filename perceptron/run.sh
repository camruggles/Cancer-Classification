echo 'These files will take roughly 11 minutes to run collectively'

echo ''
echo cross validation on ten folds
python cv.py 10

echo ''
echo cross validation on 5 folds
python cv.py 5

echo ''
echo cross validation on 2 folds
python cv.py 2


echo ''
echo 10 bootstraps
python bootstrap.py 10

echo ''
echo 5 bootstraps
python bootstrap.py 5

echo ''
echo 20 bootstraps
python bootstrap.py 20

echo 'Running leave one out cross validation'
echo 'This could take up to ten minutes'
python loocv.py
