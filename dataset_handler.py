import numpy as np
from global_variables import *

# read csv file containing dataset
def get_dataset(dataset_path):
    return np.loadtxt(dataset_path, delimiter=",", skiprows=0)

# fetch labels and features in the dataset file
def get_labels_features_from_dataset(dataset_path):
    labels = []
    features = []
    data = get_dataset(dataset_path)

    for row in data:
        labels.append(row[-1])
        features.append(tuple(row[0:-1].tolist()))

    return labels, features
