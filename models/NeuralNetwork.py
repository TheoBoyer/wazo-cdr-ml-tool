"""

    ML Model wrapper

"""

import pickle
import numpy as np
import argparse
import os
from hyperopt import hp
from hyperopt.pyll.base import scope

import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, Embedding, Dense, Dropout, BatchNormalization, SpatialDropout1D
from tensorflow.keras.models import Model as KModel, load_model

from core.MLModel import MLModel

class Model(MLModel):
    """
        Neural Network wrapper
    """
    def __init__(self, name, outputs=1):
        super(Model, self).__init__(name)
        self.categorical_features = []
        self.maps = []
        self.cp_path = "./best_model.ckpt"
        self.outputs = outputs
        self.inputs = None
        self.model = None

    def get_loss_name(self):
        raise Exception("NeuralNetwork is an abstract class. You can't call get_loss_name like that")

    def get_metrics_names(self):
        raise Exception("NeuralNetwork is an abstract class. You can't call get_metrics_names like that")

    def make_model(self, categories, n_numerical=2):
        embedding_size = int(self.get_hp("embedding_size"))
        n_layers = int(self.get_hp("n_layers"))
        n_units = int(self.get_hp("n_units"))
        dropout_rate = self.get_hp("dropout_rate")

        categorical_inputs = []
        xs = []
        for i in categories:
            placeholder = Input(shape=(1,))
            categorical_inputs.append(placeholder)
            x = Embedding(i + 1, embedding_size, input_length=1)(placeholder)
            x = SpatialDropout1D(dropout_rate)(x)
            xs.append(x[:, 0])
        numerical_input = Input(shape=(n_numerical,))
        self.inputs = categorical_inputs + [numerical_input]
        x = concatenate(xs + [numerical_input])
        x = BatchNormalization()(x)
        for i in range(n_layers):
            x = Dense(n_units, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
            x = BatchNormalization()(x)
        x = Dense(self.outputs, activation=self.get_activation_name())(x)

        model = KModel(inputs=self.inputs, outputs=x)
        #model.summary()
        return model

    def process_input(self, X):
        features = []
        for c, m in zip(self.categorical_features, self.maps):
            f = np.expand_dims(X[c].map(m).to_numpy(), -1) + 1
            f[np.isnan(f)] = 0
            features.append(f)
        numericals = X[X.columns[(X.dtypes != "category").values]].values
        numericals = (numericals - self.num_X_mean) / self.num_X_std
        features.append(numericals)
        return features

    def process(self, X, y):
        return self.process_input(X), y.values

    def build_model(self, X):
        categories = []
        for c in self.categorical_features:
            categories.append(len(X[c].unique()))
        self.model = self.make_model(categories, len(X.columns) - len(self.categorical_features))
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.get_hp("learning_rate")),
            loss=self.get_loss_name(),
            metrics=self.get_metrics_names(),
        )

    def save_norm_vals(self, X):
        numericals = X[X.columns[(X.dtypes != "category").values]].values
        self.num_X_mean = np.expand_dims(numericals.mean(axis=0), 0)
        self.num_X_std = np.expand_dims(numericals.std(axis=0), 0)

    def train(self, X, y):
        self.categorical_features = X.columns[(X.dtypes == "category")].values.tolist()
        for c in self.categorical_features:
            self.maps.append({v:i for i, v in enumerate(X[c].unique())})
        self.build_model(X)
        self.save_norm_vals(X)
        Xs, y = self.process(X, y)
        save_cp_cb = ModelCheckpoint(
            self.cp_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        self.model.fit(
            Xs, y, 
            batch_size=int(self.get_hp("batch_size")), epochs=self.get_hp("epochs"),
            validation_split=0.2, 
            shuffle=True,
            #verbose=0,
            callbacks=[
                save_cp_cb,
                early_stopping
            ]
        )

        self.model.load_weights(self.cp_path)

    def predict(self, X):
        X = self.process_input(X)
        return self.model.predict(X)

    def fetch_hp(self):
        parser = argparse.ArgumentParser(description='Neural Network')
        parser.add_argument("--embedding_size", type=int, help="Size of categorical embeddings")
        parser.add_argument("--n_layers", type=int, help="Number of layers in the neural network")
        parser.add_argument("--n_units", type=int, help="Number of units per layer in the neural network")
        parser.add_argument("--dropout_rate", type=float, help="Proportion of units to randomly shutting down in the neural network")
        parser.add_argument("--epochs", type=int, help="Number of iterations on the entire dataset")
        parser.add_argument("--batch_size", type=int, help="Number of samples to estimate the gradient")
        parser.add_argument("--learning_rate", type=float, help="Size of steps to take")
        args, _ = parser.parse_known_args()

        for k, v in vars(args).items():
            if v is not None:
                self.hp[k] = v

    def default_hp(self):
        return {
            'embedding_size': 16,
            'learning_rate': 0.0014495971065176905,
            'n_layers': 4,
            'n_units': 384,
            'dropout_rate': 0.2273963515064449,
            'epochs': 15,
            'batch_size': 108
        }

    @staticmethod
    def get_hp_search_space():
        return {
            'learning_rate': hp.loguniform('learning_rate', -15, -1.5),
            'n_units' : scope.int(hp.quniform('n_units', 64, 512, 16)),
            'n_layers' : scope.int(hp.quniform('n_layers',4,15,1)),
            'embedding_size' : scope.int(hp.quniform('embedding_size', 3, 50, 2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.5),
            'batch_size': scope.int(hp.quniform('batch_size',32,256, 1)),
            'epochs': 15
        }

    def save(self, path):
        os.mkdir(path)
        self.model.save(os.path.join(path, "model.h5"))
        with open(os.path.join(path, "wrapper"), 'wb') as f:
            pickle.dump({
                "categorical_features": self.categorical_features,
                "maps": self.maps
            }, f)
 
    @staticmethod
    def load_base(path, Model):
        model = Model()
        model.model = keras.models.load_model(os.path.join(path, "model.h5"))
        with open(os.path.join(path, "wrapper"), 'rb') as f:
            config = pickle.load(f)

        model.categorical_features = config["categorical_features"]
        model.maps = config["maps"]

        return model