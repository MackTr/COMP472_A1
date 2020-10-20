from model_handler import *

# ----- Gaussian Naive Bayes -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation datasets for the training

def run_gaussian_naive_bayes(dataset):
    gnb_model = create_model(Model.GNB)

    # train the model
    gnb_model.fit(dataset['features_train'], dataset['labels_train'])
    gnb_model.fit(dataset['features_validation'], dataset['labels_validation'])

    test_predictions_gnb = gnb_model.predict(dataset['features_test'])

    print('GND accuracy: ')
    print(get_accuracy_score(test_predictions_gnb, dataset['labels_test']))
