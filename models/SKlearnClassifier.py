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
        Xs, y = self.process(X, y)
        y_pred = self.model.predict(Xs)
        y_selected = (y_pred > 0.5).astype(np.int)
        acc = (y_selected == y).mean()
        auc_roc = roc_auc_score(y, y_pred)
        print("Logistic Regression AUC ROC:", auc_roc)
        return acc

    def predict(self, X):
        X = self.process_input(X)
        if hasattr(self.model, 'predict_proba'):
            X = self.model.predict_proba(X)
            X = X if X.shape[-1] > 2 else X[:, [1]]
        else:
            X = np.expand_dims(self.model.predict(X), -1)
        return X
