from pathlib import Path
from pprint import pprint

import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.data import StockData

BASEDIR = Path(__file__).absolute().parent.parent
MODELS_DIR = BASEDIR / 'models'
DATASDIR = BASEDIR / 'data'


class StockModel(StockData):
    def __init__(self, model, path_to_model: Path, path_to_stock_file: Path):
        self.stock = StockData(path_to_stock_file)
        self.X = self.stock[:][self.stock[:].columns.drop('CLOSE')].values
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X = torch.from_numpy(self.X).type(torch.Tensor)

        self.model = model
        self.model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        self.model.eval()

        if 'RNN' in str(self.model):
            self.X = torch.reshape(self.X, shape=[1, self.X.shape[0], self.X.shape[1]])

    def get_all_prediction(self):
        if 'RNN' in str(self.model):
            outputs, _ = self.model.forward(self.X, h_state=None)
            return outputs.detach().numpy()
        else:
            outputs = self.model.forward(self.X)
            return self.model(self.X).detach().numpy()


class RNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNNetwork, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, X, h_state):
        out, hidden_state = self.rnn(X, h_state)
        hidden_size = hidden_state[-1].size(-1)
        out = out.view(-1, hidden_size)
        outs = self.out(out)
        return outs, hidden_size
