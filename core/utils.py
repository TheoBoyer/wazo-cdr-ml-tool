"""

    Set of diverse useful functions

"""

import os
import importlib
import json
import binascii
import sys
import shutil
from distutils.dir_util import copy_tree

def get_models():
    """
        Return the list of available models
    """
    models = os.listdir("models")
    models = filter(lambda x: not x.startswith("__"), models)
    return map(lambda x: x.split('.')[0], models)

def get_tasks():
    """
        Return the list of available models
    """
    models = os.listdir("tasks")
    models = filter(lambda x: not x.startswith("__"), models)
    return map(lambda x: x.split('.')[0], models)

def load_dataset(name):
    """
        Load a dataset from its name
    """
    if name in sys.modules:
        return sys.modules[name]
    model_script = "./datasets/{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, model_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loaded_packages.append({
        "name": name,
        "path": model_script,
        "attribute": None
    })
    spec.loader.exec_module(module)
    return module

loaded_packages = []

def load_model_class(name, custom_path="./"):
    """
        Load a model class from its name
    """
    if name in sys.modules:
        return sys.modules[name].Model
    model_script = os.path.join(custom_path, "models/{}.py".format(name))
    spec = importlib.util.spec_from_file_location(name, model_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    loaded_packages.append({
        "name": name,
        "path": model_script,
        "attribute": "Model"
    })
    Model = module.Model
    return Model

def load_task_class(name, custom_path="./"):
    """
        Load a model class from its name
    """
    if name in sys.modules:
        return sys.modules[name].Model
    model_script = os.path.join(custom_path, "tasks/{}.py".format(name))
    spec = importlib.util.spec_from_file_location(name, model_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    """
    loaded_packages.append({
        "name": name,
        "path": model_script,
        "attribute": "Model"
    })
    """
    Task = getattr(module, name)
    return Task

def get_loaded_packages():
    """
        Getter for loaded packages
    """
    return loaded_packages

def get_hp(model=None):
    """
        Return the saved set of hyperparameters (by the search_hp script)
    """
    # Convert "model" to a list
    if model is None:
        model = get_models()
    if not isinstance(model, list):
        model = [model]
    hps = {}
    for m in model:
        # Load each hp
        content = None
        m_path = "hp/{}.json".format(m)
        if os.path.exists(m_path):
            with open(m_path) as f:
                content = json.load(f)
        hps[m] = content
    if len(model) > 1:
        return hps
    return hps[model[0]]

def getAvailableFolderName(base_path, folder_name):
    """
        return a folder path that doesn't already exist by appending a unique ID after it
    """
    while True:
        # Generate random names until one that is not taken is found
        random_name = str(binascii.b2a_hex(os.urandom(15)))[2:8]
        final_folder = os.path.join(base_path, folder_name + '-' + random_name)
        if not os.path.isdir(final_folder):
            break
    return final_folder

def make_output_folder(model_type):
    """
        Create a new folder for a model.
    """
    folder = getAvailableFolderName("./output/", model_type)
    trained_model_folder = os.path.join(folder, "trained_models")
    # Concatenate paths
    core_folder = os.path.join(folder, "core")
    #datasets_folder = os.path.join(folder, "datasets")
    features_folder = os.path.join(folder, "features")
    models_folder = os.path.join(folder, "models")
    config_file = os.path.join(folder, "config.json")
    # Make dirs
    os.mkdir(folder)
    os.mkdir(trained_model_folder)
    # Copy the content of the package folder into the model folder
    copy_tree("./package", folder)

    shutil.copytree("./core", core_folder)
    shutil.copytree("./models", models_folder)
    shutil.copytree("./features", features_folder)
    #shutil.copytree("./datasets", datasets_folder)

    return folder

def get_n(n_or_prop, N):
    """
        Return N if it is an integer or the closest integer of (n_or_prop * N) if it is a float number between 0 and 1
    """
    if isinstance(n_or_prop, float):
        return int(round(n_or_prop * N))
    return n_or_prop

def add_examples(data, folder, columns, n=10):
    """
        Create examples to test the  command line tools. Make them from a dataset
    """
    data = data[columns]
    for i in range(n):
        with open(os.path.join(folder, "sample{}.txt".format(i)), 'w') as f:
            sample = data.sample(1)
            for v in sample.values[0].tolist():
                f.write(str(v) + '\n')

