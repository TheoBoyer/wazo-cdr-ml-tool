"""

    Data pipeline

"""

import pandas as pd
import pickle

def requirement_order(variables):
    """
        Return an iterator that will yield the given vriables in an order such that if a variable require another, it will be yield after it
    """
    # TODO: change order by requirement
    return variables

def make_df(raw_data, variables):
    """
        Create a dataframe from the source raw data and the given variables
    """
    vars = []
    for v in requirement_order(variables):
        vars.append(v(raw_data))
    return pd.concat(vars, axis=1)

class DataPipeline:
    """
        Data pipeline
    """
    def __init__(self, features, targets):
        assert len(features), "Specify features"
        assert len(targets), "Specify targets"
        self.features = features
        self.targets = targets
        # Requirements of the configuration
        self.requirements = list(set(sum(map(lambda x: x.requirements, self.features), [])))
        self.min_header = {r: 'object' for r in self.requirements}

    def fit(self, data):
        """
            Fit the variables on a given dataset
        """
        for v in self.features + self.targets:
            v._fit(data)

    def forward(self, raw_X):
        """
            Encode a given raw column into formatted variables
        """
        return make_df(raw_X, self.features)

    def make_dataset(self):
        """
            Return the dataset formatted as variables
        """
        # Read raw data
        data = self.read_raw_data()
        self.default_header = list(data.columns.values)
        # Fit the variables on the raw dataset
        self.fit(data.copy())
        return make_df(data, self.features), make_df(data, self.targets)

    def read_raw_data(self):
        """
            Read the raw dataset (as a pandas.DataFrame)
        """
        # Must be set by the user
        raise Exception("not implemented")

    def get_input_headers(self):
        """
            Return the headers of the raw data
        """
        return [{k: 'object' for k in self.default_header}, self.min_header]

    def save(self, path):
        """
            Save the datapiepline into the given file
        """
        print("Warning: Default save used")
        with open(path, 'wb') as f:
            pickle.dump(self, f)