from core.Task import Task

class Regression(Task):
    def __init__(self, features, targets, Pipeline):
        super(Regression, self).__init__(Pipeline(features, targets), "regression")
        self.features = features
        self.targets = targets

    