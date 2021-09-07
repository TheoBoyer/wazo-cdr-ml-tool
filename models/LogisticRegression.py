from models.SKlearnClassifier import Model as SKLModel
from sklearn.linear_model import LogisticRegression
import argparse

class Model(SKLModel):
    """
        Sklearn model wrapper
    """
    def __init__(self):
        super(Model, self).__init__("LogisticRegression")

    def build_model(self):
        self.model = LogisticRegression(C=self.get_hp("C"), max_iter=250, solver='lbfgs')
    
    def fetch_hp(self):
        parser = argparse.ArgumentParser(description='Logistic regression')
        parser.add_argument("--C", type=float, help="Inverse of regularization strength")
        args, _ = parser.parse_known_args()

        for k, v in vars(args).items():
            if v is not None:
                self.hp[k] = v

    def default_hp(self):
        return {
            "C": 1.0
        }