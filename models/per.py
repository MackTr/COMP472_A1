from model_handler import *

# ----- Perceptron -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation dataset for the training

def run_perceptron(dataset):
    per_model = create_model(Model.PER)
    per_model.set_params(criterion='entropy')

    # train the model
    per_model.fit(dataset['features_train'], dataset['labels_train'])
    per_model.fit(dataset['features_validation'], dataset['labels_validation'])

    test_predictions_per = per_model.predict(dataset['features_test'])

    print('PER accuracy: ')
    print(get_accuracy_score(test_predictions_per, dataset['labels_test']))