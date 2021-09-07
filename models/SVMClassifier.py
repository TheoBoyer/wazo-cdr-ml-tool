from models.SKlearnClassifier import Model as SKLModel
from sklearn.svm import LinearSVC
import argparse

class Model(SKLModel):
    """
        Sklearn model wrapper
    """
    def __init__(self):
        super(Model, self).__init__("SVMClassifier")

    def build_model(self):
        self.model = LinearSVC(C=self.get_hp("C"))
    
    def fetch_hp(self):
        parser = argparse.ArgumentParser(description='Support vector machine classifier')
        parser.add_argument("--C", type=float, help="Inverse of regularization strength")
        args, _ = parser.parse_known_args()

        for k, v in vars(args).items():
            if v is not None:
                self.hp[k] = v

    def default_hp(self):
        return {
            "C": 1.0
        }