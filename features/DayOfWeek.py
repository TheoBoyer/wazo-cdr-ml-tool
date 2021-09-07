import pandas as pd
from core.Variable import Category

class DayOfWeek(Category):
    def __init__(self):
        super().__init__("day_of_week", ["start"])

    def forward(self, data):
        data['day_of_week'] = pd.to_datetime(data["start"]).dt.dayofweek
        return super().forward(data[['day_of_week']])

    def fit(self, data):
        data['day_of_week'] = pd.to_datetime(data["start"]).dt.dayofweek
        super().fit(data[['day_of_week']])