from models.NeuralNetwork import Model as NNModel

class Model(NNModel):
    """
       Neural Network classifier wrapper
    """
    def __init__(self, n_classes=2):
        super(Model, self).__init__("NeuralNetworkClassifier", n_classes if n_classes > 2 else 1)
        self.n_classes = n_classes

    def get_loss_name(self):
        return "binary_crossentropy" if self.n_classes <=2 else 'categorical_crossentropy'

    def get_metrics_names(self):
        return ["accuracy"]

    def get_activation_name(self):
        return 'sigmoid' if self.n_classes <=2 else 'softmax'

    @staticmethod
    def load(path):
        return NNModel.load_base(path, Model)