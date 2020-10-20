from model_handler import *

# ----- Base Decision Tree -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation datasets for the training

def run_decision_tree(dataset):
    print()
    print("------ Start: DT ------")
    dt_model = create_model(Model.DT)
    dt_model.set_params(criterion='entropy')

    # train the model
    dt_model.fit(dataset['features_train'], dataset['labels_train'])
    dt_model.fit(dataset['features_validation'], dataset['labels_validation'])

    # test the model
    test_predictions_dt = dt_model.predict(dataset['features_test'])

    # verify accuracy of the model
    print('accuracy: ')
    print(get_accuracy_score(test_predictions_dt, dataset['labels_test']))
    print("------ End: DT ------")
