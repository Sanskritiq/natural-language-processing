# CS563 NLP : Assignment 4

##### Submitted by

- Sanskriti Singh
- 2001CS60

## How to Run

- Modify the configuration as per your need in `config.yaml`
- Run following command to train and evaluate the NMT Model:
  ```bash
  python train.py
  ```

# Transformer Based NMT

## Problem Statement

- The assignment targets to implement a Neural Machine translation system with Transformers. The model needs to be trained with German as the source and English as the target

## Dataset

- Multi30K German-English parallel corpus
- Link: [Link](https://www.dropbox.com/scl/fo/jkwmeu4dkyrwm4d7bfa69/AMA9ojeZmWnkhCrWjMptueo?rlkey=q965fljtlve0fdpuvl0ofuq6p&st=1o31nn92&dl=0)
- Data format:
  - Source: zwei junge weiße männer sind im freien in der nähe vieler büsche.
  - Target: two young white males are outside near many bushes.

## Implementation

### Preprocessing

- Adding <sos> and <eos> tokens to start and end of each sentence
- Tokenize the data
- Vocabulary consists of words which are occurring more than 5 times in the entire train set

### Model

- Model is a transformer-based NMT model
  4-layers of encoder-decoder stacks, 4 attention heads
- Word embedding dimensionality is 128 and Position-wise feed forward network dimensionality is 512
- Below is the detailed model:

```
Transformer_NMT(
  (src_tok_embedding): Embedding(3515, 256)
  (src_pos_embedding): Embedding(100, 256)
  (trg_tok_embedding): Embedding(24891, 256)
  (trg_pos_embedding): Embedding(100, 256)
  (transformer): Transformer(
    (encoder): TransformerEncoder(
      (layers): ModuleList(
        (0-1): 2 x TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=512, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
      )
      (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): TransformerDecoder(
      (layers): ModuleList(
        (0-1): 2 x TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (multihead_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=512, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
        )
      )
      (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
  )
  (fc_out): Linear(in_features=256, out_features=24891, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
```

### Training

- Model was trained till convergence and perplexity of the - validation set is the stopping criteria
- CrossEntropyLoss is used as the loss function
- Below is the Adam Optimizer used:

```
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.0001
    maximize: False
    weight_decay: 0
)
```

### Configuration

- The implementation was done with the following configuration:

```yaml
dataset:
  en_train_path: Data/train.en.gz
  en_test_path: Data/test_2016_flickr.en.gz
  en_val_path: Data/val.en.gz
  de_train_path: Data/train.de.gz
  de_test_path: Data/test_2016_flickr.de.gz
  de_val_path: Data/val.de.gz
  batch_size: 32

meta:
  device: cpu

model:
  embedding_dim: 256
  n_layers: 2
  n_heads: 2
  ff_dim: 512
  max_len: 100
  dropout: 0.1
  src_pad_idx: 0
  trg_pad_idx: 0
  num_epochs: 5
```

### Results

- Wasn't able to run the code. Terminal is crashing very time.
