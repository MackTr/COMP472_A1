from model_handler import *

# ----- Best Decision Tree -----

def run_best_decision_tree(dataset):
    print()
    print("------ Start: Best-DT ------")
    best_dt_model = create_model(Model.DT)

    # create pool of parameters for the grid search
    param_grid_best_dt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, None],
        'min_samples_split': [0.00000000000001, 0.00000001, 0.0001, 0.001, 0.01, 0.1, 2, 3],
        'min_impurity_decrease': [0.00000000000001, 0.00000001, 0.0001, 0.001, 0.01, 0.1, 2, 3],
        'class_weight': [None, 'balanced']
    }

    # apply grid search on model
    best_dt_model = apply_grid_search_on(best_dt_model, param_grid_best_dt)

    # train model with training dataset print the best_params
    best_dt_model.fit(dataset['features_train'], dataset['labels_train'])
    print('This is the best parameters of Best-DT after training with training set:')
    print(best_dt_model.best_params_)

    # train model with validation dataset and print the best_params
    best_dt_model.fit(dataset['features_validation'], dataset['labels_validation'])
    print('This is the best parameters of Best-DT after training with validation set:')
    print(best_dt_model.best_params_)

    # test the model
    test_predictions_best_dt = best_dt_model.predict(dataset['features_test'])

    # verify accuracy of the model
    print(' accuracy: ')
    print(get_accuracy_score(test_predictions_best_dt, dataset['labels_test']))
    print("------ End: Best-DT ------")
