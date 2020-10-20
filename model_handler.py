from enum import Enum

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

# type of models
class Model(Enum):
    GNB = 1
    DT = 2
    PER = 3
    MLP = 4

# create a model according to the given parameter
def create_model(model_type):
    if model_type == Model.GNB:
        return GaussianNB()
    elif model_type == Model.DT:
        return DecisionTreeClassifier()
    elif model_type == Model.PER:
        return Perceptron()
    elif model_type == Model.MLP:
        return MLPClassifier()
    else:
        return None


# applying the given param_grid to the model using GridSearch
def apply_grid_search_on(model, param_grid):
    return GridSearchCV(model, param_grid)

# returns accuracy score of a test
def get_accuracy_score(test_predictions, right_labels):
    return metrics.accuracy_score(test_predictions, right_labels)
