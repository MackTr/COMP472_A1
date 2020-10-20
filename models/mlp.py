from model_handler import *

# ----- Base Multi-Layered Perceptron -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation dataset for the training

def run_multi_layer_perceptron(dataset):
    mlp_model = create_model(Model.MLP)

    # train the model
    mlp_model.fit(dataset['features_train'], dataset['labels_train'])
    mlp_model.fit(dataset['features_validation'], dataset['labels_validation'])

    test_predictions_mlp = mlp_model.predict(dataset['features_test'])

    print('MLP accuracy: ')
    print(get_accuracy_score(test_predictions_mlp, dataset['labels_test']))