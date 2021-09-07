"""

    ML Model wrapper

"""

import pickle


class MLModel:
    """
        Machine Learning wrapper
    """
    def __init__(self, name):
        self.name = name
        self.hp = {}
        self.fetch_hp()
        self.default = self.default_hp()
        self.fill_hp_with_default()

    def fill_hp_with_default(self):
        """
            Fill-in missing hp with d-their default value
        """
        for k, v in self.default.items():
            if k not in self.hp:
                self.hp[k] = v

    def get_hp(self, hp_name):
        """
            Return a given hp
        """
        assert hp_name in self.hp, "{} not provided and not found in default hp config".format(hp_name)
        return self.hp[hp_name]

    def set_hp(self, hp):
        """
            Sette for the hp set
        """
        self.hp = hp
        self.fill_hp_with_default()

    def __str__(self):
        return "{}({})".format(self.name.title(), self.hp)

    def train(self, X, y):
        """
            Abstract method to train the model
        """
        raise Exception("not implemented")

    def evaluate(self, X, y):
        """
            Abstract method to evaluate the model
        """
        raise Exception("not implemented")

    def predict(self, X):
        """
            Abstract method to make predictions
        """
        raise Exception("not implemented")

    def save(self, path):
        """
            Save the model as a pickle object
        """
        print("Warning: Default save used")
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
            Load a model from a pickle object
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def fetch_hp(self):
        """
            Fetch hp from command line arguments
        """
        print("Warning: fetch_hp not implemented")
        pass

    def default_hp(self):
        """
            return default values of each hp
        """
        print("Warning: default_hp not implemented")
        return {}

    @staticmethod
    def get_hp_search_space():
        """
            return an hyperopt search space for hp optimization
        """
        raise Exception("not implemented")
        