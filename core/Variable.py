"""

    Wrapper for a variable

"""

from core.utils import get_n
from collections import defaultdict

def assert_serie_valid(x, name):
    """
        Assert that the given serie do not contain NaN and is either a numerical or categorical variable
    """
    assert not x.isna().any(), "NaN(s) were found in variable " + name
    assert "float" in x.dtype.name or x.dtype.name == "category", "Each variable must be categorical or numerical, not {}".format(x.dtype)

class Variable:
    """
        Variable Wrapper
    """
    def __init__(self, name, requirements):
        self.name = name
        self.requirements = requirements
        self.fitted = False

    def _fit(self, data):
        """
            Fit the variable on the given data
        """
        self.fitted = True
        self.fit(data[self.requirements].copy())

    def __call__(self, raw):
        """
            Transform a raw dataset into a variable
        """
        if not self.fitted:
            raise Exception("Variable not fitted")
        x = self.forward(raw[self.requirements].copy())
        assert_serie_valid(x, self.name)
        return x

    def forward(self, data):
        """
            Transform raw data into a variable
        """
        raise Exception("not implemented")

    def fit(self, data):
        """
            Fit the variable on the given data
        """
        pass

class Numerical(Variable):
    def __init__(self, name, requirements=[], interpolation='mean'):
        super().__init__(name, requirements if len(requirements) else [name])
        self.interpolation = interpolation

    def forward(self, data):
        raw = data[self.name].astype(float)
        raw = raw.fillna(self.default_value)
        return raw.astype(float)

    def fit(self, data):
        raw = data[self.name].astype(float)
        if self.interpolation == 'mean':
            self.default_value = raw[~(raw.isna())].mean()
        else:
            raise ValueError("Unkown interpolation method:", self.interpolation)

def int_cat():
    return -1

def obect_cat():
    return 'rare'

class Category(Variable):
    def __init__(self, name, requirements=[], min_class_samples=25):
        super().__init__(name, requirements if len(requirements) else [name])
        self.min_class_samples = min_class_samples

    def forward(self, data):
        raw = data[self.name]
        raw = self.normalize(raw)
        
        raw = raw.map(self.categories)

        return raw.astype("category")

    def normalize(self, column):
        return column.fillna("nan").astype(str)

    def fit(self, data):
        self.min_class_samples = get_n(self.min_class_samples, len(data))
        raw = data[self.name]
        raw = self.normalize(raw)

        vcounts = raw.value_counts()
        if raw.dtype =='int':
            self.categories = defaultdict(int_cat)
        else:
            self.categories = defaultdict(obect_cat)
        self.categories.update({v: v for v in vcounts[vcounts >= self.min_class_samples].index.values})


class Binary(Category):
    """
        Binary Variable
    """
    def __init__(self, name, requirements=[], default=0):
        super().__init__(name, requirements if len(requirements) else [name])
        self.default = default

    def forward(self, data):
        raw = data[self.name]
        raw = raw.fillna(self.default)
        return raw.astype(float)

    def fit(self, data):
        """
            Fit the variable on the given data
        """
        pass