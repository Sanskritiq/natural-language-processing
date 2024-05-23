import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from torch.nn import CrossEntropyLoss

from tqdm import tqdm

from data.preprocess.dataloader import get_dataloader
from models.models import get_model

import pandas as pd
import numpy as np

from yaml import load, Loader
config = load(open('config.yaml'), Loader=Loader)

dataset_types = ['binary']
model_types = ['rnn', 'ffnn']


def train_on_diff_configs():
    accuracy_df = pd.DataFrame()
    for dataset_type in dataset_types:
        for model_type in model_types:
            print(f'Training {model_type} model on {dataset_type} dataset...')
            model = get_model(dataset_type, model_type)
            if dataset_type == 'binary':
                # criterion = BCELoss()
                criterion = CrossEntropyLoss()
            elif dataset_type == 'multiclass':
                criterion = CrossEntropyLoss()
            optimizer = Adam(model.parameters(), lr=config['training']['lr'])
            train_dataloader, test_dataloader = get_dataloader(dataset_type)
            
            for epoch in tqdm(range(config['training']['num_epochs']), position = 0, leave=False):
                model.train()
                for i, (data, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), position = 1, leave=False):
                    # print(data.shape)
                    optimizer.zero_grad()
                    outputs = model(data)
                    if(dataset_type == 'binary'):
                        outputs = outputs.squeeze()
                    loss = criterion((outputs), labels)  # Squeeze the outputs if necessary
                    loss.backward()
                    optimizer.step()
                    if i % 100 == 0:
                        print(f'\nEpoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
                
                model.eval()
                with torch.no_grad():
                    total, correct = 0, 0
                    for i, (data, labels) in enumerate(test_dataloader):
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    print(f'Epoch: {epoch}, Accuracy: {100*correct/total}\n')
                    if epoch==config['training']['num_epochs']-1:
                        # accuracy_df.append({'Dataset Type': dataset_type, 'Model Type': model_type, 'Accuracy': 100*correct/total}, ignore_index=True)
                        accuracy_df = pd.concat([accuracy_df, pd.DataFrame({'Dataset Type': dataset_type, 'Model Type': model_type, 'Accuracy': 100*correct/total}, index=[0])], ignore_index=True)
    print(accuracy_df)
                    
if __name__ == '__main__':
    train_on_diff_configs()

