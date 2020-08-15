from decimal import Decimal
from pprint import pprint
from typing import Dict

from torch import nn

from src.data import StockData
from src.models import DATASDIR, MODELS_DIR, StockModel


class BackTrader(StockModel, StockData):
    def __init__(self, model, path_to_model, path_to_stock_file):
        self.model = StockModel(model, path_to_model, path_to_stock_file)
        self.stock = StockData(path_to_stock_file)

        self.predict = self.model.get_all_prediction()

    def _get_length_stock(self) -> int:
        return self.stock[:].shape[0]

    def buy(self, i: int, count: int) -> Decimal:
        if self.predict[i] > self.predict[i + 1]:
            return count * Decimal(self.stock[i]['OPEN'])

    def sell(self, i: int, count: int) -> Decimal:
        if self.predict[i] < self.predict[i + 1]:
            return count * Decimal(self.stock[i]['OPEN'])

    def close_deal(self, i: int, type: str, count: int) -> Dict[str, Decimal]:
        close = Decimal(self.stock[i]['CLOSE']) * count
        if type == 'buy':
            open = self.buy(i, count)
            return {'type': 'buy', 'open': open, 'close': close, 'profit': close - open}
        elif type == 'sell':
            open = self.sell(i, count)
            return {'type': 'close', 'open': open, 'close': close, 'profit': open - close}

    def get_all_backtrading(self, count):
        backtrading = []
        for i in range(self._get_length_stock()):
            buy = self.buy(i, count)
            sell = self.sell(i, count)
            if buy:
                close_deal = self.close_deal(i, type='buy', count=count)
                backtrading.append(close_deal)
            elif sell:
                close_deal = self.close_deal(i, type='sell', count=count)
                backtrading.append(close_deal)
        return backtrading

    def get_profit(self, count):
        profit = [0]
        backtrading = self.get_all_backtrading(count)
        for i in range(len(backtrading)):
            profit.append(profit[i] + backtrading[i]['profit'])
        return profit


class BuyHold(StockData):
    def __init__(self, path_to_stock_file):
        self.stock = StockData(path_to_stock_file)

    def _get_length_stock(self) -> int:
        return self.stock[:].shape[0]

    def buy(self, i: int, count: int) -> Decimal:
        return Decimal(self.stock[i]['OPEN'] * count)

    def sell(self, i: int, count: int) -> Decimal:
        return Decimal(self.stock[i]['OPEN'] * count)

    def close_deal(self, i: int, type: str, count: int) -> Dict[str, Decimal]:
        close = Decimal(self.stock[i]['CLOSE']) * count
        if type == 'buy':
            open = self.buy(i, count)
            return {'type': 'buy', 'open': open, 'close': close, 'profit': close - open}
        elif type == 'sell':
            open = self.sell(i, count)
            return {'type': 'close', 'open': open, 'close': close, 'profit': open - close}

    def get_all_backtrading(self, count):
        backtrading = []
        for i in range(self._get_length_stock()):
            buy = self.buy(i, count)
            if buy:
                close_deal = self.close_deal(i, type='buy', count=count)
                backtrading.append(close_deal)
        return backtrading

    def get_profit(self, count):
        profit = [0]
        backtrading = self.get_all_backtrading(count)
        for i in range(len(backtrading)):
            profit.append(profit[i] + backtrading[i]['profit'])
        return profit
