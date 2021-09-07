from models.NeuralNetwork import Model as NNModel
import tensorflow as tf

class Model(NNModel):
    """
       Neural Network regressor wrapper
    """
    def __init__(self, n_outputs=1):
        super(Model, self).__init__("NeuralNetworkRegressor", n_outputs)

    def get_loss_name(self):
        return "mean_squared_error"

    def get_metrics_names(self):
        return [tf.keras.metrics.RootMeanSquaredError()]

    def get_activation_name(self):
        return 'linear'

    @staticmethod
    def load(path):
        return NNModel.load_base(path, Model)