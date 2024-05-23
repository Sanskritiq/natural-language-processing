import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_preprocess import load_data, preprocess_data

import yaml
config = yaml.safe_load(open("config.yaml"))

batch_size = config['dataset']['batch_size']

class NMT_Dataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.trg_data[idx]
    
def collate_fn(batch):
    src_data, trg_data = zip(*batch)
    return src_data, trg_data

def get_dataloaders():
    raw_data = load_data()
    processed_data, vocab = preprocess_data(raw_data)
    
    src_vocab_size = len(vocab[0])
    trg_vocab_size = len(vocab[1])
    
    train_data = NMT_Dataset(processed_data[0][0], processed_data[1][0])
    test_data = NMT_Dataset(processed_data[0][1], processed_data[1][1])
    val_data = NMT_Dataset(processed_data[0][2], processed_data[1][2])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader, val_loader, src_vocab_size, trg_vocab_size

if __name__ == '__main__':
    train_loader, test_loader, val_loader, src_vocab_size, trg_vocab_size = get_dataloaders()
    
    for batch_idx, batch in enumerate(train_loader):
        print(batch)
        print("batch-en----------------------")
        print(batch[0][0])
        print("batch-de----------------------")
        print(batch[0][1])
        
        print("batch-de----------------------")
        
        print(len(batch[0]), len(batch[1]), type(batch[0]), type(batch[1]))
        
        break