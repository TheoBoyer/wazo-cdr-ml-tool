import pandas as pd
from core.Variable import Variable

class TimeOfDay(Variable):
    def __init__(self):
        super().__init__("time_of_day", ["start"])

    def forward(self, data):
        data['start'] = pd.to_datetime(data["start"])
        time_of_day = (3600 * data['start'].dt.hour + 60 * data['start'].dt.minute + data['start'].dt.second) / 3600 / 24
        return time_of_day