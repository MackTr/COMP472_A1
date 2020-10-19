from model_handler import *
from dataset_handler import *

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



# ----- Gaussian Naive Bayes -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation datasets for the training

gnb_model = create_model(Model.GNB)

# train the model
gnb_model.fit(features_train, labels_train)
gnb_model.fit(features_validation, labels_validation)

test_predictions_gnb = gnb_model.predict(features_test)

print('GND accuracy: ')
print(get_accuracy_score(test_predictions_gnb, labels_validation))



# ----- Base Decision Tree -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation datasets for the training

dt_model = create_model(Model.DT)
dt_model.set_params(criterion='entropy')

# train the model
dt_model.fit(features_train, labels_train)
dt_model.fit(features_validation, labels_validation)

test_predictions_dt = dt_model.predict(features_test)

print('DT accuracy: ')
print(get_accuracy_score(test_predictions_dt, labels_validation))



# ----- Best Decision Tree -----

best_dt_model = create_model(Model.DT)

param_grid_best_dt = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, None],
    'min_samples_split': [0.00000000000001, 0.00000001, 0.0001, 0.1],
    'min_impurity_decrease': [0.00000000000001, 0.00000001, 0.0001, 0.1],
    'class_weight': [None, 'balanced']
}

best_dt_model = apply_grid_search_on(best_dt_model, param_grid_best_dt)

# train model with training dataset print the best_params
best_dt_model.fit(features_train, labels_train)
print(best_dt_model.best_params_)

# train model with validation dataset and print the best_params
best_dt_model.fit(features_validation, labels_validation)
print(best_dt_model.best_params_)

test_predictions_best_dt = best_dt_model.predict(features_test)

print('Best-DT accuracy: ')
print(get_accuracy_score(test_predictions_best_dt, labels_validation))




# ----- Perceptron -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation dataset for the training

per_model = create_model(Model.PER)
per_model.set_params(criterion='entropy')

# train the model
per_model.fit(features_train, labels_train)
per_model.fit(features_validation, labels_validation)

test_predictions_per = per_model.predict(features_test)

print('PER accuracy: ')
print(get_accuracy_score(test_predictions_per, labels_validation))




# ----- Base Multi-Layered Perceptron -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation dataset for the training

mlp_model = create_model(Model.MLP)

# train the model
mlp_model.fit(features_train, labels_train)
mlp_model.fit(features_validation, labels_validation)

test_predictions_mlp = mlp_model.predict(features_test)

print('MLP accuracy: ')
print(get_accuracy_score(test_predictions_mlp, labels_validation))



# ----- Best Multi-Layered Perceptron -----

best_mlp_model = create_model(Model.MLP)

param_grid_best_mlp = {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'hidden_layer_sizes': [(30, 50), (20, 20, 20), (10, 20, 30)],
    'solver': ['adam', 'sgd']
}

best_mlp_model = apply_grid_search_on(best_mlp_model, param_grid_best_mlp)

# train model with training dataset print the best_params
best_mlp_model.fit(features_train, labels_train)
print(best_dt_model.best_params_)

# train model with validation dataset and print the best_params
best_mlp_model.fit(features_validation, labels_validation)
print(best_dt_model.best_params_)

test_predictions_best_dt = best_mlp_model.predict(features_test)

print('Best-MLP accuracy: ')
print(get_accuracy_score(test_predictions_best_dt, labels_validation))
