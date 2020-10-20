from model_handler import *
from sklearn.metrics import confusion_matrix
import numpy as np
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
    
    # print the row of the instance, followed by a comma, and then its associated predicted class by the model
    for number in range(len(test_predictions_per)):
        print(int(test_predictions_per[number]),int(dataset['labels_test'][number]),sep=',')

    print("##################PERCEPTRON CONFUSION MATRIX############################")

    print(confusion_matrix(test_predictions_per, dataset['labels_test']))

    print("PERCEPTRON precision") 
    print(metrics.precision_score(test_predictions_per, dataset['labels_test'], average=None, zero_division=1))

    print("PERCEPTRON recall")
    print(metrics.recall_score(test_predictions_per, dataset['labels_test'], average=None, zero_division=1))

    print("PERCEPTRON f1 by class")
    print(metrics.f1_score(test_predictions_per, dataset['labels_test'], average=None))

    print("PERCEPTRON accuracy")
    print(metrics.accuracy_score(test_predictions_per, dataset['labels_test']))

    print("PERCEPTRON macro average f1")
    print(metrics.f1_score(test_predictions_per, dataset['labels_test'], average='macro'))

    print("PERCEPTRON weighted average f1")
    print(metrics.f1_score(test_predictions_per, dataset['labels_test'], average='weighted'))

    print("------ End: PER ------")

    ###Write to csv######

    #If do not want to produce csv file, change print to False

    Print = True

    if (Print ==True):
        #Change file name for second dataset if it is the case
        f = open("CSV/PER-DS2.csv", "w")
        f.write("PERCEPTRON\n")
        # print the row of the instance, followed by a comma, and then its associated predicted class by the model
        for number in range(len(test_predictions_per)):
            preprocessingPred = int(test_predictions_per[number])
            proprocessingTest = int(dataset['labels_test'][number])
            f.write(str(preprocessingPred))
            f.write(",")
            f.write(str(proprocessingTest))
            f.write("\n")

        f.write("##################CONFUSION MATRIX############################\n")
        f.write(np.array2string(confusion_matrix(test_predictions_per, dataset['labels_test'])))

        f.write("\nPrecision\n")
        f.write(np.array2string(metrics.precision_score(test_predictions_per, dataset['labels_test'], average=None, zero_division=1)))

        f.write("\nRecall\n")
        f.write(np.array2string(metrics.recall_score(test_predictions_per, dataset['labels_test'], average=None, zero_division=1)))

        f.write("\nF1 by class\n")
        f.write(np.array2string(metrics.f1_score(test_predictions_per, dataset['labels_test'], average=None)))

        f.write("\nAccuracy\n")
        f.write(str(metrics.accuracy_score(test_predictions_per, dataset['labels_test'])))

        f.write("\nMacro average F1\n")
        f.write(str(metrics.f1_score(test_predictions_per, dataset['labels_test'], average='macro')))

        f.write("\nWeighted average F1\n")
        f.write(str(metrics.f1_score(test_predictions_per, dataset['labels_test'], average='weighted')))
