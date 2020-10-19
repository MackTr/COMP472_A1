import numpy as np
from helper import *
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

i1 = 'Assig1-Dataset/Assig1-Dataset/info_1.csv'
tn1 = 'Assig1-Dataset/Assig1-Dataset/test_no_label_1.csv'
tw1 = 'Assig1-Dataset/Assig1-Dataset/test_with_label_1.csv'
t1 = 'Assig1-Dataset/Assig1-Dataset/train_1.csv'
v1 = 'Assig1-Dataset/Assig1-Dataset/val_1.csv'

i2 = 'Assig1-Dataset/Assig1-Dataset/info_2.csv'
tn2 = 'Assig1-Dataset/Assig1-Dataset/test_no_label_2.csv'
tw2 = 'Assig1-Dataset/Assig1-Dataset/test_with_label_2.csv'
t2 = 'Assig1-Dataset/Assig1-Dataset/train_2.csv'
v2 = 'Assig1-Dataset/Assig1-Dataset/val_2.csv'


labels_train = []
features_train = []

labels_test = []
features_test = []

data_train = np.loadtxt(t1, delimiter=",", skiprows=0)
data_test = np.loadtxt(tw1, delimiter=",", skiprows=0)

labels_train, features_train = get_labels_features(data_train)
labels_test, features_test = get_labels_features(data_test)

model = GaussianNB()

model.fit(features_train, labels_train)

test_predictions = model.predict(features_test)

for number in range(len(test_predictions)):
    print(int(test_predictions[number]),int(labels_test[number]),sep=',')

print(confusion_matrix(test_predictions, labels_test))

print("precision")
print(metrics.precision_score(test_predictions, labels_test, average=None, zero_division=1))

print("recall")
print(metrics.recall_score(test_predictions, labels_test, average=None, zero_division=1))

print("f1 by class")
print(metrics.f1_score(test_predictions, labels_test, average=None))

print("accuracy")
print(metrics.accuracy_score(test_predictions, labels_test))

print("macro average f1")
print(metrics.f1_score(test_predictions, labels_test, average='macro'))

print("weighted average f1")
print(metrics.f1_score(test_predictions, labels_test, average='weighted'))
