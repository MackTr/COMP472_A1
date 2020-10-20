from model_handler import *

# ----- Best Multi-Layered Perceptron -----

def run_best_multi_layered_perceptron(dataset):
    best_mlp_model = create_model(Model.MLP)

    param_grid_best_mlp = {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(30, 50), (20, 20, 20), (10, 20, 30)],
        'solver': ['adam', 'sgd']
    }

    best_mlp_model = apply_grid_search_on(best_mlp_model, param_grid_best_mlp)

    # train model with training dataset print the best_params
    best_mlp_model.fit(dataset['features_train'], dataset['labels_train'])
    print(best_mlp_model.best_params_)

    # train model with validation dataset and print the best_params
    best_mlp_model.fit(dataset['features_validation'],dataset['labels_validation'])
    print(best_mlp_model.best_params_)

    test_predictions_best_dt = best_mlp_model.predict(dataset['features_test'])

    print('Best-MLP accuracy: ')
    print(get_accuracy_score(test_predictions_best_dt, dataset['labels_test']))