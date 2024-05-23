import torch 
import torch.nn as nn
import torch.optim as optim

from data_preprocess import load_data, preprocess_data
from dataloader import get_dataloaders
from model import Transformer_NMT

from tqdm import tqdm

import yaml
config = yaml.safe_load(open("config.yaml"))

print('configuration:', config, sep='\n')

train_loader, test_loader, val_loader, src_vocab_size, trg_vocab_size = get_dataloaders()

device = config['meta']['device']
embedding_dim = config['model']["embedding_dim"]
n_heads = config['model']['n_heads']
n_layers = config['model']['n_layers']
src_pad_idx = config['model']['src_pad_idx']
trg_pad_idx = config['model']['trg_pad_idx']
ff_dim = config['model']['ff_dim']
max_len = config['model']['max_len']
dropout = config['model']['dropout']
epoch = config['model']['num_epochs']

model = Transformer_NMT(
    embedding_dim = embedding_dim,
    src_vocab_size = src_vocab_size,
    trg_vocab_size = trg_vocab_size,
    n_heads = n_heads,
    n_layers = n_layers,
    src_pad_idx = src_pad_idx,
    ff_dim = ff_dim,
    max_len = max_len,
    dropout = dropout,
    device = device,
).to(device)

print("model:", model, sep='\n')

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

print("optimizer:", optimizer, sep='\n')
print("criterion:", criterion, sep='\n')

# train function
train_losses = []
eval_losses = []

for i in range(epoch):
    # training
    model.train()
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress"):
        in_src = torch.tensor(batch[0]).to(device)
        out_trg = torch.tensor(batch[1]).to(device)

        output = model(in_src, out_trg)
        # output shape: [trg_seq_len, batch_size, trg_vocab_size]
        output = output.permute(1, 2, 0)  # Swap batch_size and trg_vocab_size dimensions
        # output shape: [batch_size, trg_vocab_size, trg_seq_len]
        target = out_trg 
        # target shape: [batch_size, trg_seq_len - 1]

        output_dim = output.shape[1]
        output = output.contiguous().view(-1, output_dim)
        # output shape: [(trg_seq_len - 1) * batch_size, trg_vocab_size]
        target = target.contiguous().view(-1)
        # target shape: [(trg_seq_len - 1) * batch_size]
        print('output:', output, output.shape)
        print('target:', target, target.shape)

        optimizer.zero_grad()
        train_loss = criterion(output, target)
        print(f'train_loss: {train_loss}')
        train_losses.append(train_loss)

        train_loss.backward()
        optimizer.step()
    
    # eval
    model.eval()
    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluation Progress"):
        in_src = torch.tensor(batch[0]).to(device)
        out_trg = torch.tensor(batch[1]).to(device)

        output = model(in_src, out_trg)
        # output shape: [trg_seq_len, batch_size, trg_vocab_size]
        output = output.permute(1, 2, 0)  # Swap batch_size and trg_vocab_size dimensions
        # output shape: [batch_size, trg_vocab_size, trg_seq_len]
        target = out_trg 
        # target shape: [batch_size, trg_seq_len - 1]

        output_dim = output.shape[1]
        output = output.contiguous().view(-1, output_dim)
        # output shape: [(trg_seq_len - 1) * batch_size, trg_vocab_size]
        target = target.contiguous().view(-1)
        # target shape: [(trg_seq_len - 1) * batch_size]

        eval_loss = criterion(output, target)
        eval_losses.append(eval_loss)

    print(f'Epoch: {i+1}/{epoch}')
    print(f'Training Loss: {sum(train_losses)/len(train_losses):,.3f}\tEvaluation Loss: {sum(eval_losses)/len(eval_losses):,.3f}')
    print(f'Training PPL: {math.exp(sum(train_losses)/len(train_losses)):,.3f}\tEvaluation PPL: {math.exp(sum(eval_losses)/len(eval_losses)):,.3f}')