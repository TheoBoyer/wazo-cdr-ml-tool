"""

    Cross-validation wrapper

"""

import numpy as np
import os
import json

from core.TMPFolder import TMPFolder
from core.utils import load_model_class, make_output_folder, get_loaded_packages, add_examples, load_task_class
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, log_loss, mean_squared_error

# Metrics for classification
CLASSIFICATION_METRICS = {
    "Cross-entropy": log_loss,
    "F1": lambda y_true, y_pred: f1_score(y_true, (y_pred > 0.5).astype(np.int)),
    "AUC ROC": roc_auc_score,
    "Accuracy": lambda y_true, y_pred: accuracy_score(y_true, (y_pred > 0.5).astype(np.int))
}
# Metrics for regression
REGRESSION_METRICS = {
    "mse": mean_squared_error
}
# Metrics to measure during cross-validation
METRICS = {
    "classification": CLASSIFICATION_METRICS,
    "regression": REGRESSION_METRICS,
    "none": {}
}

class CrossValidation:
    """
        Cross-validation wrapper
    """
    def __init__(self, task, cv=5):
        Task = load_task_class(task)
        self.task = Task()
        self.cv = cv
        self.model_save = []

    def train(self, model_type, params=None):
        """
            Train the given model on the given dataset
        """
        Model = load_model_class(model_type)
        self.model_type = model_type
        X, y = self.task.make_dataset()
        self.final_data = X.copy()
        # Save preds
        preds = np.zeros_like(y.values).astype(np.float)
        with TMPFolder():
            N = len(X)
            n = N // self.cv
            # Assign a fold to each sample
            folds = np.random.permutation(np.repeat(np.arange(self.cv), n+1)[:N])
            if self.cv == 1:
                folds[:] = 1
                folds[np.random.permutation(np.arange(N))[:int(round(0.25 * N))]] = 0
            # Iterate over folds
            for k in range(self.cv):
                print("Fold", k)
                # Create model
                model = Model()
                if params is not None:
                    model.set_hp(params)
                # Create sub-dataset
                X_train = X[folds != k]
                y_train = y[folds != k]
                X_test = X[folds == k]
                y_test = y[folds == k]
                # Train the model
                model.train(X_train, y_train)
                # Make predictions on test samples
                y_pred = model.predict(X_test)
                # Save the predictions
                preds[folds == k] = y_pred
                self.model_save.append(model)
            # Save folds
            self.folds = folds
        self.is_trained = True
        self.preds = preds
        self.true_labels = y

    def get_trained_model(self, Model, X, y, params=None):
        """
            Train a model and return it
        """
        assert self.is_trained, "You need to train the models before getting them"
        return self.model_save

    def get_val_metrics(self, metrics_type="classification"):
        """
            Train a model and return validation metrics obtained
        """
        assert self.is_trained, "You need to train the models before getting validation metrics"
        # Get the set of metrics adapted to the task
        metrics = METRICS[metrics_type]
        # Train the model
        preds = self.preds
        score = {}
        for k, v in metrics.items():
            score[k] = v(self.true_labels.values, preds)
        return score

    def save(self, path):
        """
            Save the models obtained by cross-validation
        """
        for i, m in enumerate(self.model_save):
            m.save(os.path.join(path, str(i) + "-" + m.name))

    def make_package(self):
        # creates a folder with an id and setup the base file structure
        folder = make_output_folder(self.model_type)

        # Create the dataset
        print("Loading dataset...")
        
        # Train, perform cross validation and returns metrics
        print("Training with cross validation...")
        metrics = self.get_val_metrics(metrics_type=self.task.metric_type)
        print("Final metrics: ", metrics)

        print("Making package...")

        config = {
            # Type of model
            "model_type": self.model_type,
            # Packages required
            "packages": get_loaded_packages()
        }
        # Saves the config file
        with open(os.path.join(folder, "config.json"), 'w') as f:
            json.dump(config, f)

        # Saves the metrics of the cross validation training
        with open(os.path.join(folder, "metrics.json"), 'w') as f:
            json.dump(metrics, f)

        # Saves the validation predictions, labels and folds
        self.final_data["preds"] = self.preds
        self.final_data["labels"] = self.true_labels
        self.final_data["folds"] = self.folds
        # Saves everything
        self.final_data.to_csv(os.path.join(folder, "validation_preds.csv"), index=False)

        # Create a bunch of examples to test the model later
        add_examples(self.task.read_raw_data(), os.path.join(folder, "examples"), self.task.get_requirements())

        # Saves the pipeline
        self.task.save_pipeine(os.path.join(folder, "pipeline.pkl"))
        # Saves the trained models
        self.save(os.path.join(folder, "trained_models"))
