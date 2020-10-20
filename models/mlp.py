from model_handler import *
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
# ----- Base Multi-Layered Perceptron -----
# Since we don't need to find the best parameters for this model
# we will use the train and validation dataset for the training

def run_multi_layer_perceptron(dataset):
    print()
    print("------ Start: Base-MLP ------")
    mlp_model = create_model(Model.MLP)
    mlp_model.set_params(hidden_layer_sizes=100, activation='logistic', solver='sgd')

    # train the model
    mlp_model.fit(dataset['features_train'], dataset['labels_train'])
    mlp_model.fit(dataset['features_validation'], dataset['labels_validation'])

    # test the model
    test_predictions_mlp = mlp_model.predict(dataset['features_test'])

    # print the row of the instance, followed by a comma, and then its associated predicted class by the model
    for number in range(len(test_predictions_mlp)):
        print(int(test_predictions_mlp[number]),int(dataset['labels_test'][number]),sep=',')

    print("##################Base Multi-Layered Perceptron CONFUSION MATRIX############################")

    print(confusion_matrix(test_predictions_mlp, dataset['labels_test']))

    print("Base Multi-Layered Perceptron precision") 
    print(metrics.precision_score(test_predictions_mlp, dataset['labels_test'], average=None, zero_division=1))

    print("Base Multi-Layered Perceptron recall")
    print(metrics.recall_score(test_predictions_mlp, dataset['labels_test'], average=None, zero_division=1))

    print("Base Multi-Layered Perceptron f1 by class")
    print(metrics.f1_score(test_predictions_mlp, dataset['labels_test'], average=None))

    print("Base Multi-Layered Perceptron accuracy")
    print(metrics.accuracy_score(test_predictions_mlp, dataset['labels_test']))

    print("Base Multi-Layered Perceptron macro average f1")
    print(metrics.f1_score(test_predictions_mlp, dataset['labels_test'], average='macro'))

    print("Base Multi-Layered Perceptron weighted average f1")
    print(metrics.f1_score(test_predictions_mlp, dataset['labels_test'], average='weighted'))

    print("------ End: MLP ------")

    ###Write to csv######

    #If do not want to produce csv file, change print to False

    Print = True

    if (Print ==True):
        #Change file name for second dataset if it is the case
        f = open("CSV/Base-MLP-DS2.csv", "w")
        f.write("Base Multi-Layered Perceptron\n")
        # print the row of the instance, followed by a comma, and then its associated predicted class by the model
        for number in range(len(test_predictions_mlp)):
            preprocessingPred = int(test_predictions_mlp[number])
            proprocessingTest = int(dataset['labels_test'][number])
            f.write(str(preprocessingPred))
            f.write(",")
            f.write(str(proprocessingTest))
            f.write("\n")

        f.write("##################CONFUSION MATRIX############################\n")
        f.write(np.array2string(confusion_matrix(test_predictions_mlp, dataset['labels_test'])))

        f.write("\nPrecision\n")
        f.write(np.array2string(metrics.precision_score(test_predictions_mlp, dataset['labels_test'], average=None, zero_division=1)))

        f.write("\nRecall\n")
        f.write(np.array2string(metrics.recall_score(test_predictions_mlp, dataset['labels_test'], average=None, zero_division=1)))

        f.write("\nF1 by class\n")
        f.write(np.array2string(metrics.f1_score(test_predictions_mlp, dataset['labels_test'], average=None)))

        f.write("\nAccuracy\n")
        f.write(str(metrics.accuracy_score(test_predictions_mlp, dataset['labels_test'])))

        f.write("\nMacro average F1\n")
        f.write(str(metrics.f1_score(test_predictions_mlp, dataset['labels_test'], average='macro')))

        f.write("\nWeighted average F1\n")
        f.write(str(metrics.f1_score(test_predictions_mlp, dataset['labels_test'], average='weighted')))