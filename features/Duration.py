import pandas as pd
import numpy as np
from core.Variable import Numerical

class Duration(Numerical):
    def __init__(self):
        super().__init__("duration", ["duration", "start", "end"])

    def forward(self, data):
        data['duration'] = (pd.to_datetime(data["end"]) - pd.to_datetime(data["start"])).dt.total_seconds()
        data['duration'] = np.log(data['duration'].values)
        return super().forward(data[['duration']])

    def fit(self, data):
        data['duration'] = (pd.to_datetime(data["end"]) - pd.to_datetime(data["start"])).dt.total_seconds()
        super().fit(data[['duration']])