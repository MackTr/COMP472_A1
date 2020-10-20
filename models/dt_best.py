from model_handler import *
from sklearn.metrics import confusion_matrix
import numpy as np
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

    # print the row of the instance, followed by a comma, and then its associated predicted class by the model
    for number in range(len(test_predictions_best_dt)):
        print(int(test_predictions_best_dt[number]),int(dataset['labels_test'][number]),sep=',')

    print("##################Best DT CONFUSION MATRIX############################")

    print(confusion_matrix(test_predictions_best_dt, dataset['labels_test']))

    print("Best DT precision") 
    print(metrics.precision_score(test_predictions_best_dt, dataset['labels_test'], average=None, zero_division=1))

    print("Best DT recall")
    print(metrics.recall_score(test_predictions_best_dt, dataset['labels_test'], average=None, zero_division=1))

    print("Best DT f1 by class")
    print(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average=None))

    print("Best DT accuracy")
    print(metrics.accuracy_score(test_predictions_best_dt, dataset['labels_test']))

    print("Best DT macro average f1")
    print(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average='macro'))

    print("Best DT weighted average f1")
    print(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average='weighted'))

    print("------ End: Best-DT ------")

    ###Write to csv######

    #If do not want to produce csv file, change print to False

    Print = True

    if (Print ==True):
        #Change file name for second dataset if it is the case
        f = open("CSV/Best-DT-DS2.csv", "w")
        f.write("Best Decision Tree\n")
        # print the row of the instance, followed by a comma, and then its associated predicted class by the model
        for number in range(len(test_predictions_best_dt)):
            preprocessingPred = int(test_predictions_best_dt[number])
            proprocessingTest = int(dataset['labels_test'][number])
            f.write(str(preprocessingPred))
            f.write(",")
            f.write(str(proprocessingTest))
            f.write("\n")

        f.write("##################CONFUSION MATRIX############################\n")
        f.write(np.array2string(confusion_matrix(test_predictions_best_dt, dataset['labels_test'])))

        f.write("\nPrecision\n")
        f.write(np.array2string(metrics.precision_score(test_predictions_best_dt, dataset['labels_test'], average=None, zero_division=1)))

        f.write("\nRecall\n")
        f.write(np.array2string(metrics.recall_score(test_predictions_best_dt, dataset['labels_test'], average=None, zero_division=1)))

        f.write("\nF1 by class\n")
        f.write(np.array2string(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average=None)))

        f.write("\nAccuracy\n")
        f.write(str(metrics.accuracy_score(test_predictions_best_dt, dataset['labels_test'])))

        f.write("\nMacro average F1\n")
        f.write(str(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average='macro')))

        f.write("\nWeighted average F1\n")
        f.write(str(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average='weighted')))
