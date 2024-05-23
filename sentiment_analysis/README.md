# CS563 NLP | Assignment 3

#### Submitted by:
- Sanskriti Singh
- 2001CS60

## Problem Statement

 - To implement Feed-Forward NN and RNN for Binary and Multi-class sentiment analysis

## How to Run

- Go to the main directory
- Download the dataset
- add dataset path in the `config.yaml`
- Run `preprocess.py` to preprocess and store the data

    ```bash
    python data/preprocess/preprocess.py
    ```
- Run `train.py` to train the model on various datasets and models

    ```bash
    python train.py
    ```
- Conditions of the training and preprocessing can be adjusted through `config.yaml`

## Dataset

- There were two types of dataset:
    - Binary
    - Multiclass

### Preprocessing

- The preprocessing involved:
    - Removing special characters
    - Removing HTML tags
    - Removing URLS
    - Removing digits
    - Removing stopping words
    - Reducing words to stem words
    - Tokenizing
    - Conversion to vector

```text
Loading saved data...
Binary Dataset:

Number of training samples: 25000
Number of test samples: 25000
Batch size: 64
Number of train batches: 391
Number of test batches: 391

Loading saved data...
Multiclass Dataset:

Number of training samples: 120000
Number of test samples: 7600
Batch size: 64
Number of train batches: 1875
Number of test batches: 119
```

## Models

### Structure

 - Two models were used for the classification:
    - RNN
    - Feed-Forward NN

```text
BinaryRNNModel(
  (lstm): LSTM(3000, 256, batch_first=True)
  (fc): Linear(in_features=256, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
BinaryFFNNModel(
  (fc1): Linear(in_features=3000, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)
MultiClassRNNModel(
  (lstm): LSTM(3000, 256, batch_first=True)
  (fc): Linear(in_features=256, out_features=4, bias=True)
)
MultiClassFFNNModel(
  (fc1): Linear(in_features=3000, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
)
```

## Results

```text
  Dataset Type Model Type   Accuracy                                                                                                                  
0       binary        rnn  50.000000
1       binary       ffnn  50.000000
2   multiclass        rnn  89.671053
3   multiclass       ffnn  87.552632

```
