import argparse
import os
import importlib
import warnings
warnings.filterwarnings("ignore")

def get_plugins():
    models = os.listdir("plugins")
    models = filter(lambda x: not x.startswith("__"), models)
    return map(lambda x: x.split('.')[0], models)

def load_plugin(name):
    model_script = "./plugins/{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, model_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main

def main():
    parser = argparse.ArgumentParser(description='Run a plugin in the package')
    parser.add_argument("plugin_name", choices=list(get_plugins()), help="Name of the plugin to run")
    args, _ = parser.parse_known_args()

    plugin = load_plugin(args.plugin_name)
    plugin()


if __name__ == "__main__":
    main()