"""

    ML Model wrapper

"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from core.MLModel import MLModel

class Model(MLModel):
    """
        Sklearn model wrapper
    """
    def __init__(self, name):
        super(Model, self).__init__(name)

    def build_model(self):
        raise Exception("build_model not implemented")

    def process_input(self, X):
        one_hot = self.oh_enc.transform(X[self.categorical_features]).toarray()
        X = X.drop(self.categorical_features, axis=1).values
        X = np.concatenate([X, one_hot], axis=1)
        return X

    def process(self, X, y):
        return self.process_input(X), y.values

    def train(self, X, y):
        self.build_model()
        self.oh_enc = OneHotEncoder(handle_unknown='ignore')
        #self.oh_enc = OneHotEncoder()
        self.categorical_features = X.columns[(X.dtypes == "category")].values.tolist()
        self.oh_enc.fit(X[self.categorical_features])
        X, y = self.process(X, y)
        self.model.fit(X, y)
