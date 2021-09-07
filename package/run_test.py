import argparse
import os
import importlib
import warnings
warnings.filterwarnings("ignore")

def get_tests():
    models = os.listdir("tests")
    models = filter(lambda x: not x.startswith("__"), models)
    return map(lambda x: x.split('.')[0], models)

def load_test(name):
    model_script = "./tests/{}.py".format(name)
    spec = importlib.util.spec_from_file_location(name, model_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run

def main():
    parser = argparse.ArgumentParser(description='Run a test in the package')
    parser.add_argument("test_name", choices=list(get_tests()), help="Name of the test to run")
    args, _ = parser.parse_known_args()

    test = load_test(args.test_name)
    test()


if __name__ == "__main__":
    main()