from model_handler import *

# ----- Perceptron -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation dataset for the training

def run_perceptron(dataset):
    print()
    print("------ Start: PER ------")
    per_model = create_model(Model.PER)

    # train the model
    per_model.fit(dataset['features_train'], dataset['labels_train'])
    per_model.fit(dataset['features_validation'], dataset['labels_validation'])

    # test the model
    test_predictions_per = per_model.predict(dataset['features_test'])

    # verify accuracy of the model
    print('PER accuracy: ')
    print(get_accuracy_score(test_predictions_per, dataset['labels_test']))
    print("------ End: PER ------")
