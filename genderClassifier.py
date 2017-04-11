from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
import numpy as np

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
logi_clf = linear_model.LogisticRegression()
svm_clf= svm.SVC()
NB_clf= naive_bayes.GaussianNB()
# 3

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data

# 3 steps for each classifier -train, predict and check accuracy

clf = clf.fit(X, Y)
tree_pred = clf.predict(X)
acc_tree = accuracy_score(Y, tree_pred)*100

logi_clf = logi_clf.fit(X, Y)
logi_pred =logi_clf.predict(X)
acc_logi = accuracy_score(Y, logi_pred)*100

svm_clf = svm_clf.fit(X, Y)
svm_pred =svm_clf.predict(X)
acc_svm = accuracy_score(Y, svm_pred)*100

NB_clf = NB_clf.fit(X, Y)
NB_pred = NB_clf.predict(X)
acc_NB = accuracy_score(Y, NB_pred)*100

# CHALLENGE compare their reusults and print the best one!

index = np.argmax([acc_logi, acc_svm, acc_NB])
classifier = {0: "logistic", 1:"SVM", 2:"GaussianNB"}

print("The Best Classifier is", classifier[index])
