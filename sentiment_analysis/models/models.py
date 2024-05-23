import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from yaml import load, Loader
config = load(open('config.yaml'), Loader=Loader)

import torch
import torch.nn as nn


# RNN Model using LSTM input = 1024, hidden = 256, output = 1
class BinaryRNNModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, output_size=1):
        super(BinaryRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
    
# RNN Model using LSTM input = 1024, hidden = 256, output = 4
class MultiClassRNNModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=256, output_size=4):
        super(MultiClassRNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
# FFNN Model input = 1024, hidden_size1 = 256, hidden_size2 = 128, output = 1
class BinaryFFNNModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size1=256, hidden_size2=128, output_size=1):
        super(BinaryFFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out
    
# FFNN Model input = 1024, hidden_size1 = 256, hidden_size2 = 128, output = 4
class MultiClassFFNNModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size1=256, hidden_size2=128, output_size=4):
        super(MultiClassFFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
def get_model(dataset_type:str='binary', model_type:str='rnn') -> nn.Module:
    
    if dataset_type=='binary':
        if model_type=='rnn':
            input_size = config['dataset']['binary']['max_features']
            hidden_size = config['model']['rnn']['hidden_size']
            output_size = 1
            return BinaryRNNModel(input_size, hidden_size, output_size)
        elif model_type=='ffnn':
            input_size = config['dataset']['binary']['max_features']
            hidden_size_1 = config['model']['ffnn']['hidden_size_1']
            hidden_size_2 = config['model']['ffnn']['hidden_size_2']
            output_size = 1
            return BinaryFFNNModel(input_size, hidden_size_1, hidden_size_2, output_size)
        else:
            raise ValueError('Invalid model type')
    elif dataset_type=='multiclass':
        if model_type=='rnn':
            input_size = config['dataset']['multiclass']['max_features']
            hidden_size = config['model']['rnn']['hidden_size']
            output_size = config['dataset']['multiclass']['num_classes']
            return MultiClassRNNModel(input_size, hidden_size, output_size)
        elif model_type=='ffnn':
            input_size = config['dataset']['multiclass']['max_features']
            hidden_size_1 = config['model']['ffnn']['hidden_size_1']
            hidden_size_2 = config['model']['ffnn']['hidden_size_2']
            output_size = config['dataset']['multiclass']['num_classes']
            return MultiClassFFNNModel(input_size, hidden_size_1, hidden_size_2, output_size)
        else:
            raise ValueError('Invalid model type')
    else:
        raise ValueError('Invalid dataset type')
    
if __name__ == '__main__':
    model = get_model('binary', 'rnn')
    print(model)
    model = get_model('binary', 'ffnn')
    print(model)
    model = get_model('multiclass', 'rnn')
    print(model)
    model = get_model('multiclass', 'ffnn')
    print(model)