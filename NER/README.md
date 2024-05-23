# CS563-NLP | Assignment 2
Submitted by - 
- Sanskriti Singh
- 2001CS60

## Problem Statement

- The assignment targets to implement Hidden Markov Model (HMM) to perform Part-of-Speech (PoS) tagging task

## Data

- Dataset consists of sentences and each word is tagged with its corresponding PoS tag
- Brown dataset: `NER-Dataset-Train.txt`
- Format of dataset:
    - Each line contains `<Word \t Tag>` (word followed by tab-space and tag)
    - Sentences are separated by a new line

### Preprocessing

- Data was prepared by separating `tokens` and `tags`

## HMM Model

- HMM model had various parts, as discussed below

### States

- Bigram
    - Single tags are used as the `states`

- Trigram
    - Pair of tags are used as `states`

### Observations

- total `tokens` observed in train set is the `observations`

### Start Probability

- Calculation of the probability of each `state` to occur at the start of the sentence
- A linear array of probabilities for each `state`

### Transition Probability

- Calculation of probability of moving from one `state` to another
- A 2D matrix containing probabilities of States VS States


### Emission Probability

- Probability of an `observation` being a particular `state`
- A 2D matrix containing Probabilities of Observations VS States

## Testing and Results

- Model was tested using 5 Fold Cross Validation Method 

### Bigram

- Accuracy of the model was calculated for each testing set

#### Metrics

```
Bigram
  0%|                                                                                                                                  | 0/5 [00:00<?, ?it/s]
1
Accuracy: 0.9950941271570357
Precision: 0.9967376858798427
Recall: 0.9950941271570357
F1: 0.9949935633269285
 20%|████████████████████████▍                                                                                                 | 1/5 [00:00<00:01,  2.08it/s]
2
Accuracy: 0.9975725200725202
Precision: 0.9975690082644628
Recall: 0.9975725200725202
F1: 0.9973919234537693
 40%|████████████████████████████████████████████████▊                                                                         | 2/5 [00:00<00:01,  2.14it/s]
3
Accuracy: 0.9968227914061246
Precision: 0.9967425973297184
Recall: 0.9968227914061246
F1: 0.9964204511844265
 60%|█████████████████████████████████████████████████████████████████████████▏                                                | 3/5 [00:01<00:00,  2.17it/s]
4
Accuracy: 0.9990327226868446
Precision: 0.9996469270840487
Recall: 0.9990327226868446
F1: 0.9992703905508559
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 4/5 [00:01<00:00,  2.18it/s]
5
Accuracy: 0.9934311886891118
Precision: 0.9965624853929581
Recall: 0.9934311886891118
F1: 0.9945465730554345
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.17it/s]
```

### Trigram

#### Metrics

```
Trigram
  0%|                                                                                                                                  | 0/5 [00:00<?, ?it/s]Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1: 1.0
 20%|████████████████████████▍                                                                                                 | 1/5 [00:00<00:02,  1.47it/s]Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1: 1.0
 40%|████████████████████████████████████████████████▊                                                                         | 2/5 [00:01<00:02,  1.45it/s]Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1: 1.0
 60%|█████████████████████████████████████████████████████████████████████████▏                                                | 3/5 [00:02<00:01,  1.46it/s]Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1: 1.0
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 4/5 [00:02<00:00,  1.43it/s]
Accuracy: 0.9988826815642459
Precision: 0.9997206703910614
Recall: 0.9988826815642459
F1: 0.999185288640596
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:03<00:00,  1.43it/s]
```
