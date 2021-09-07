from core.Task import Task

class Classification(Task):
    def __init__(self, features, targets, Pipeline):
        super(Classification, self).__init__(Pipeline(features, targets), "classification")
        self.features = features
        self.targets = targets

    