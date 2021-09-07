import pandas as pd
from core.Variable import Category

class Hour(Category):
    def __init__(self):
        super().__init__("hour", ["start"])

    def forward(self, data):
        data['hour'] = pd.to_datetime(data["start"]).dt.hour
        return super().forward(data[['hour']])

    def fit(self, data):
        data['hour'] = pd.to_datetime(data["start"]).dt.hour
        super().fit(data[['hour']])