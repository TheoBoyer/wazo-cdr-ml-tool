import pandas as pd

from core.DataPipeline import DataPipeline

class CDR(DataPipeline):
    def read_raw_data(self):
        data = pd.read_csv("./data/cdr.csv")
        #return data
        return data.loc[data["stack"] == 2]

    def format_input(self, data):
        data = data.sort_values("start")
        data = data[self.requirements]
        return data

if __name__ == "__main__":
    cdr = CDR()
    X, y = cdr.make_dataset()

    print(X.dtypes)
    print(y.dtypes)