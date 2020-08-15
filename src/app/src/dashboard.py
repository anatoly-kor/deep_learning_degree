from decimal import Decimal
from pathlib import Path
from pprint import pprint

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from dash.dependencies import Input, Output

from src.backtrading import BackTrader, BuyHold
from src.models import RNNNetwork, StockModel

BASEDIR = Path(__file__).absolute().parent.parent
DATASDIR = BASEDIR / 'data'
MODELS_DIR = BASEDIR.parent.parent / 'models'

STOCK_FILE = DATASDIR / 'GAZP_200424_200725.csv'


model_rnn = RNNNetwork(4, 256, 1, 1)
path_to_model_rnn = MODELS_DIR / 'RNN_Adam.pt'
stock_model_rnn = StockModel(model_rnn, path_to_model_rnn, STOCK_FILE)
stock_rnn = stock_model_rnn.stock[:].reset_index()
stock_rnn['DATE'] = pd.to_datetime(stock_rnn[['Year', 'Month', 'Day']])

predict_rnn = [float(i) for i in stock_model_rnn.get_all_prediction()]


model_rnn_sgd = RNNNetwork(4, 256, 1, 1)
path_to_model_rnn_sgd = MODELS_DIR / 'RNN_SGD.pt'
stock_model_rnn_sgd = StockModel(model_rnn_sgd, path_to_model_rnn_sgd, STOCK_FILE)
stock_rnn_sgd = stock_model_rnn_sgd.stock[:].reset_index()
stock_rnn_sgd['DATE'] = pd.to_datetime(stock_rnn_sgd[['Year', 'Month', 'Day']])

predict_rnn_sgd = [float(i) for i in stock_model_rnn_sgd.get_all_prediction()]


lot = 1
count = 10 * lot

back_trader_rnn = BackTrader(model_rnn, path_to_model_rnn, STOCK_FILE)
profit_rnn = back_trader_rnn.get_profit(count)

back_trader_rnn_sgd = BackTrader(model_rnn_sgd, path_to_model_rnn_sgd, STOCK_FILE)
profit_rnn_sgd = back_trader_rnn_sgd.get_profit(count)


back_trader_buy_hold = BuyHold(STOCK_FILE)
profit_buy_hold = back_trader_buy_hold.get_profit(count)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dash_table.DataTable(
        id='stock-data-table',
        page_size=20,
        columns=[{'name': i, 'id': i} for i in stock_rnn.columns],
        data=stock_rnn.to_dict('records')
    ),
    html.Div([
        html.Button(
            'New Day',
            id='get-new-day',
            n_clicks=0
        ),
        dcc.Graph(
            id='stock-data-graph',
        )
    ], style={'marginTop': '30px'}),

    html.Div([
        dcc.Graph(
            id='backtrader-graph',
            figure={
                'data': [
                    {'x': [i for i in range(len(profit_rnn))],
                     'y': profit_rnn, 'name': 'Profit RNN Adam'},
                    {'x': [i for i in range(len(profit_rnn_sgd))],
                     'y': profit_rnn_sgd, 'name': 'Profit RNN SGD'},
                    {'x': [i for i in range(len(profit_buy_hold))],
                     'y': profit_buy_hold, 'name': 'Profit Buy&Hold'}
                ],
                'layout': {
                    'title': 'Backtesting'
                }
            }
        )
    ])
])


predict_rnn_list = []
predict_rnn_sgd_list = []


@app.callback(
    Output('stock-data-graph', 'figure'),
    [Input('get-new-day', 'n_clicks')]
)
def update_stock_graph(n_clicks):

    predict_rnn_list.append(predict_rnn[n_clicks])

    predict_rnn_sgd_list.append(predict_rnn_sgd[n_clicks])

    return {
        'data': [
            {'x': stock_rnn['DATE'], 'y': stock_rnn['CLOSE'], 'name': 'Real Stock'},
            {'x': stock_rnn['DATE'], 'y': predict_rnn_list, 'name': 'RNN_Adam Predict'},
            {'x': stock_rnn['DATE'], 'y': predict_rnn_sgd_list, 'name': 'RNN_SGD Predict'}
        ]
    }


if __name__ == "__main__":
    app.run_server(debug=True)
