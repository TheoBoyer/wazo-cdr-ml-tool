"""

    Train a machine learning model and saves it.

"""

import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")

from core.CrossValidation import CrossValidation, METRICS
from core.utils import get_models, get_tasks, load_task_class

def main():
    models = get_models()
    tasks = get_tasks()
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a machine learning model and saves it.')
    parser.add_argument("model_type", choices=list(models), help="Size of categorical embeddings")
    parser.add_argument("task", choices=list(tasks), help="Task to train a model on")
    parser.add_argument("--cv", type=int, default=5, help="Nomber of folds for cross validation")
    args, _ = parser.parse_known_args()

    cv = CrossValidation(args.task, args.cv)
    cv.train(args.model_type)
    cv.make_package()

if __name__ == "__main__":
    main()