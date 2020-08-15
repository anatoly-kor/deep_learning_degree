from pathlib import Path

import pandas as pd
from pandas import DataFrame


class StockData:
    def __init__(self, path_to_file: Path):
        self.data = pd.read_csv(path_to_file)
        self.data = self.data.drop(columns=['TIME'])
        self.data['DATE'] = pd.to_datetime(self.data['DATE'], format='%Y%m%d')
        self.data['Year'] = self.data['DATE'].dt.year
        self.data['Month'] = self.data['DATE'].dt.month
        self.data['Day'] = self.data['DATE'].dt.day

        self.data = self.data[self.data.columns.drop('DATE')]
        self.data = self.data.set_index(['Year', 'Month', 'Day'])

    def __getitem__(self, i):
        return self.data.iloc[i]
