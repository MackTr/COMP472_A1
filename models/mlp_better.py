from model_handler import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
# ----- Best Multi-Layered Perceptron -----

def run_best_multi_layered_perceptron(dataset):
    print()
    print("------ Start: Best-MLP ------")
    best_mlp_model = create_model(Model.MLP)

    # create pool of parameters for the grid search
    param_grid_best_mlp = {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'hidden_layer_sizes': [(30, 50), (20, 20, 20)],
        'solver': ['adam', 'sgd']
    }

    # apply grid search on model
    best_mlp_model = apply_grid_search_on(best_mlp_model, param_grid_best_mlp)

    # train model with training dataset print the best_params
    best_mlp_model.fit(dataset['features_train'], dataset['labels_train'])
    print('This is the best parameters of Best_MLP after training with training set:')
    print(best_mlp_model.best_params_)

    # train model with validation dataset and print the best_params
    best_mlp_model.fit(dataset['features_validation'],dataset['labels_validation'])
    print('This is the best parameters of Best_MLP after training with validation set:')
    print(best_mlp_model.best_params_)

    # test the model
    test_predictions_best_dt = best_mlp_model.predict(dataset['features_test'])    

    # print the row of the instance, followed by a comma, and then its associated predicted class by the model
    for number in range(len(test_predictions_best_dt)):
        print(int(test_predictions_best_dt[number]),int(dataset['labels_test'][number]),sep=',')

    print("##################Best-MLP CONFUSION MATRIX############################")

    print(confusion_matrix(test_predictions_best_dt, dataset['labels_test']))

    print("Best-MLP precision") 
    print(metrics.precision_score(test_predictions_best_dt, dataset['labels_test'], average=None, zero_division=1))

    print("Best-MLP recall")
    print(metrics.recall_score(test_predictions_best_dt, dataset['labels_test'], average=None, zero_division=1))

    print("Best-MLP f1 by class")
    print(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average=None))

    print("Best-MLP accuracy")
    print(metrics.accuracy_score(test_predictions_best_dt, dataset['labels_test']))

    print("Best-MLP macro average f1")
    print(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average='macro'))

    print("Best-MLP weighted average f1")
    print(metrics.f1_score(test_predictions_best_dt, dataset['labels_test'], average='weighted'))

    print("------ End: Best-MLP ------")

    ###Write to csv######

    #If do not want to produce csv file, change print to False

    Print = True

    if (Print ==True):
        #Change file name for second dataset if it is the case
        f = open("CSV/Best-MLP-DS2.csv", "w")
        f.write("Best Multi-Layered Perceptron\n")
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

