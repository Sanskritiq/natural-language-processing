'''
Preprocessing:
- Adding <sos> and <eos> tokens to start and end of each sentence
- Tokenize the data
- Vocabulary consists of words which are occurring more than 5 times (in case of en) in the entire train set

'''

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

import yaml

config = yaml.safe_load(open("config.yaml"))

def load_data():
    print('loading data...', flush=True)
    # load .gz file, dont use comma as separator, no header
    en_train_data = pd.read_csv(config['dataset']['en_train_path'], compression='gzip', encoding='utf-8', sep='\t', header=None)
    en_train_data.columns = ['sentence']
    en_test_data = pd.read_csv(config['dataset']['en_test_path'], compression='gzip', encoding='utf-8', sep='\t', header=None)
    en_test_data.columns = ['sentence']
    en_val_data = pd.read_csv(config['dataset']['en_val_path'], compression='gzip', encoding='utf-8', sep='\t', header=None)
    en_val_data.columns = ['sentence']
    
    de_train_data = pd.read_csv(config['dataset']['de_train_path'], compression='gzip', encoding='utf-8', sep='\t', header=None)
    de_train_data.columns = ['sentence']
    de_test_data = pd.read_csv(config['dataset']['de_test_path'], compression='gzip', encoding='utf-8', sep='\t', header=None)
    de_test_data.columns = ['sentence']
    de_val_data = pd.read_csv(config['dataset']['de_val_path'], compression='gzip', encoding='utf-8', sep='\t', header=None)
    de_val_data.columns = ['sentence']
    
    print('data loaded')    
    return (en_train_data, en_test_data, en_val_data), (de_train_data, de_test_data, de_val_data)

def remove_empty(data):
    for lang_data in data:
        for type_data in lang_data:
            print('removing empty rows...', flush=True)
            type_data.dropna(inplace=True)
    
    return data

def tokenize(data):
    
    new_data = []
    
    for lang_data in data:
        new_lang_data = []
        for type_data in lang_data:
            print('tokenizing...', flush=True)
            type_data = list(type_data['sentence'].apply(lambda x: x.split()))
            new_lang_data.append(type_data)
        new_data.append(new_lang_data)
            
    print('tokenization done')
    
    return new_data

def create_vocab(data):
    vocab = []
    for lang_data in data:
        lang_vocab = {}
        type_data = lang_data[0]
        print('creating vocab...', flush=True)
        for sentence in type_data:
            for word in sentence:
                if word in lang_vocab:
                    lang_vocab[word] += 1
                else:
                    lang_vocab[word] = 1
        vocab.append(lang_vocab)
    
    vocab[0] = {k: v for k, v in vocab[0].items() if v > 5}
    new_vocab = []
    for lang_vocab in vocab:
        lang_vocab = list(lang_vocab.keys())
        lang_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + lang_vocab
        new_vocab.append(lang_vocab)
    
    print('vocab created')
    
    return new_vocab

def add_padding(data):
    
    new_data = []
    max_len = 0
    
    for lang_data in data:
        new_lang_data = []
        for type_data in lang_data:
            print('adding end padding...', flush=True)
            type_data = list(map(lambda x: ['<sos>'] + x + ['<eos>'], type_data))
            max_len = max(max_len, max(map(len, type_data)))
            new_lang_data.append(type_data)
        new_data.append(new_lang_data)
    
    print('end padding added')
    
    print('max_len:', max_len)
    
    new_new_data = []
    for lang_data in new_data:
        new_new_lang_data = []
        for type_data in lang_data:
            print('adding length padding...', flush=True)
            new_type_data = []
            for sentence in type_data:
                new_sentence = sentence + ['<pad>'] * (max_len - len(sentence))
                new_type_data.append(new_sentence)
            new_new_lang_data.append(new_type_data)
        new_new_data.append(new_new_lang_data)
        
    print('length padding added')    
    
    return new_new_data

def add_unknown(data, vocab):
    
    new_data = []
    for lang_data, lang_vocab in zip(data, vocab):
        new_lang_data = []
        for type_data in lang_data:
            print('adding unknown...', flush=True)
            type_data = list(map(lambda x: [word if word in lang_vocab else '<unk>' for word in x], type_data))
            new_lang_data.append(type_data)
        new_data.append(new_lang_data)
    
    print('unknown added')
    
    return new_data

def replace_with_index(data, vocab):
        
    new_data = []
    for lang_data, lang_vocab in zip(data, vocab):
        new_lang_data = []
        for type_data in lang_data:
            print('replacing with index...', flush=True)
            type_data = list(map(lambda x: [lang_vocab.index(word) for word in x], type_data))
            new_lang_data.append(type_data)
        new_data.append(new_lang_data)
    
    print('replaced with index')
    
    return new_data

def preprocess_data(data):
    data = remove_empty(data)
    data = tokenize(data)
    vocab = create_vocab(data)
    data = add_padding(data)
    data = add_unknown(data, vocab)
    data = replace_with_index(data, vocab)
    
    return data, vocab


if __name__ == '__main__':
    en_data, de_data = load_data()
    
    print("raw-train-----------------------")
    
    print(en_data[0].head())
    print(de_data[0].head())
    
    print("raw-test-----------------------")
    
    print(en_data[1].head())
    print(de_data[1].head())
    
    print("raw-val-----------------------")
    
    print(en_data[2].head())
    print(de_data[2].head())
    
    processed_data, vocab = preprocess_data([en_data, de_data])
    
    print("------------------------")
    print("pros-en-train----------------------")
    
    print(processed_data[0][0][:5])
    
    print("pros-en-test----------------------")
    
    print(processed_data[0][1][:5])
    
    print("pros-en-val----------------------")
    
    print(processed_data[0][2][:5])
    
    print("pros-de-train----------------------")
    
    print(processed_data[1][0][:5])
    
    print("pros-de-test----------------------")
    
    print(processed_data[1][1][:5])
    
    print("pros-de-val----------------------")
    
    print(processed_data[1][2][:5])
    
    print("vocab------------------------")
    
    print(vocab[0][:10])
    print(vocab[1][:10])
    
    print("padding_index:", vocab[0][0], vocab[1][0])
    