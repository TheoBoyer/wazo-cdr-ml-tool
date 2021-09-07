"""

    ML Model wrapper

"""

import numpy as np
from sklearn.metrics import roc_auc_score
from models.SKlearnModel import Model as SKLModel


class Model(SKLModel):
    """
        Sklearn Classifier model wrapper
    """
    def __init__(self, name):
        super(Model, self).__init__(name)

    def evaluate(self, X, y):
        raise Exception("Not implemented")

    def predict(self, X):
        raise Exception("Not implemented")
