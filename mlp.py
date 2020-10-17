import numpy as np
from helper import *
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

i1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/info_1.csv'
tn1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/test_no_label_1.csv'
tw1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/test_with_label_1.csv'
t1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/train_1.csv'
v1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/val_1.csv'

i2 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/info_2.csv'
tn2 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/test_no_label_2.csv'
tw2 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/test_with_label_2.csv'
t2 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/train_2.csv'
v2 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/val_2.csv'


labels_train = []
features_train = []

labels_test = []
features_test = []

data_train = np.loadtxt(t1, delimiter=",", skiprows=0)
data_test = np.loadtxt(tw1, delimiter=",", skiprows=0)

labels_train, features_train = get_labels_features(data_train)
labels_test, features_test = get_labels_features(data_test)

model = MLPClassifier(hidden_layer_sizes=100, activation='logistic', solver='sgd')

model.fit(features_train, labels_train)

test_predictions = model.predict(features_test)

for number in range(len(test_predictions)):
    if test_predictions[number] == labels_test[number]:
        print('woohoo')
    else:
        print('fck')

print(metrics.accuracy_score(test_predictions, labels_test))