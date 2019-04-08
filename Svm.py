Code for SVM

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Zoo.csv')
X = dataset.iloc[:, 1: 17].values
y = dataset.iloc[:, 17].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
from sklearn.model_selection import cross_val_score
from sklearn import svm
clf = svm.SVC(kernel='sigmoid', C=1)
scores = cross_val_score(clf, X_test,y_test, cv=5)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report
target_names = ['Class-1','Class-2', 'Class-3 ','Class-4','Class-5','Class-6','Class-7']
print(classification_report(y_test, y_pred, target_names=target_names))

