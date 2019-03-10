from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

gridParams = {
    'C': (2, 5),
    #'gamma': (0.001, 1)
}


def get_est_params():
    return gridParams


def create_est(kwargs):
    return LogisticRegression(C=kwargs.get("C"), random_state=17, solver='liblinear')

