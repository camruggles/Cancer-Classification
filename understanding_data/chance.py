# Malignant is denoted by 1 and Benign as -1 in the original dataset
# Converting benign to 0 

import read_clean

X,y = read_clean.getCleanedData("data.csv")
y = [0 if x == -1 else x for x in y]

total = len(y)
malignant =  sum(y)
benign = total - malignant

print(total, malignant, benign)

# Classifier that predicts Benign always
print(float(malignant)/total)

# Classifier that predicts malignant always
print(float(benign)/total)
