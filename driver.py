from model_handler import *
from dataset_handler import *

from models.gnb import *
from models.dt import *
from models.dt_best import *
from models.per import *
from models.mlp import *
from models.mlp_better import *

# load datasets from csv files
labels_train = []
features_train = []

labels_test = []
features_test = []

labels_validation = []
features_validation = []

labels_train, features_train = get_labels_features_from_dataset(train_dataset_path)
labels_test, features_test = get_labels_features_from_dataset(test_with_label_path)
labels_validation, features_validation = get_labels_features_from_dataset(validation_dataset_path)

dataset = {
    'labels_train': labels_train,
    'features_train': features_train,
    'labels_test': labels_test,
    'features_test': features_test,
    'labels_validation': labels_validation,
    'features_validation': features_validation,
}

run_gaussian_naive_bayes(dataset)
run_decision_tree(dataset)
run_best_decision_tree(dataset)
run_perceptron(dataset)
run_multi_layer_perceptron(dataset)
run_best_multi_layered_perceptron(dataset)
