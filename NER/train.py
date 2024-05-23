import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import HMMModel, RNNModel
from preprocess import preprocess, get_bigram_probabilities, get_trigram_probabilities, process_data_bigram, process_data_trigram
from evaluate import Evaluate

FILEPATH = 'NER-Dataset-Train.txt'

# N cross validation on HMM, RNN
class NCrossValidation:
    def __init__(self, sentence_token_tag, n=5, method='bigram'):
        self.sentence_token_tag = sentence_token_tag
        self.n = n
        self.method = method
        
    def run(self):
        
        self.part_size = len(self.sentence_token_tag) // self.n
        for rnd in tqdm(range(1, self.n+1)):
            overall_accuracy = 0
            overall_precision = 0
            overall_recall = 0
            overall_f1 = 0
            overall_confusion_matrix = 0
            
            test_set = self.sentence_token_tag[self.part_size * (rnd-1) : self.part_size * rnd]
            train_set = [sentence for sentence in self.sentence_token_tag if sentence not in test_set]
            test_pred, test_ground = [], []
            
            if self.method=='bigram':
                train_set = self.sentence_token_tag
                states, observations, transition_prob, emission_prob, start_prob = get_bigram_probabilities(self.sentence_token_tag)
                test_states, test_observations, test_ground = process_data_bigram(test_set, states)
                
                gram_model = HMMModel(states, observations, transition_prob, emission_prob, start_prob)
                
                for i in range(len(test_set)):
                    obs = test_observations[i]
                    true = test_ground[i]
                    bestpath, bestpath_prob = gram_model.viterbi_new(obs)
                    test_pred.append(bestpath)
                    evaluate = Evaluate(true, bestpath)
                    overall_accuracy += evaluate.accuracy
                    overall_precision += evaluate.precision
                    overall_recall += evaluate.recall
                    overall_f1 += evaluate.f1
                    
                overall_accuracy /= len(test_set)
                overall_precision /= len(test_set)
                overall_recall /= len(test_set)
                overall_f1 /= len(test_set)
                
                print(rnd)
                print(f'Accuracy: {overall_accuracy}')
                print(f'Precision: {overall_precision}')
                print(f'Recall: {overall_recall}')
                print(f'F1: {overall_f1}')
                
                
            if self.method=='trigram':
                train_set = self.sentence_token_tag
                test_set = [sentence for sentence in test_set if len(sentence) > 1]
                states, observations, transition_prob, emission_prob, start_prob = get_trigram_probabilities(self.sentence_token_tag)
                test_states, test_observations, test_ground = process_data_trigram(test_set, states)
                
                gram_model = HMMModel(states, observations, transition_prob, emission_prob, start_prob)
                
                for i in range(len(test_set)):
                    obs = test_observations[i]
                    true = test_ground[i]
                    bestpath, bestpath_prob = gram_model.viterbi_new(obs)
                    test_pred.append(bestpath)
                    evaluate = Evaluate(true, bestpath)
                    overall_accuracy += evaluate.accuracy
                    overall_precision += evaluate.precision
                    overall_recall += evaluate.recall
                    overall_f1 += evaluate.f1
                    
                overall_accuracy /= len(test_set)
                overall_precision /= len(test_set)
                overall_recall /= len(test_set)
                overall_f1 /= len(test_set)
                
                print(f'Accuracy: {overall_accuracy}')
                print(f'Precision: {overall_precision}')
                print(f'Recall: {overall_recall}')
                print(f'F1: {overall_f1}')
                    
            if self.method=='rnn':
                pass
            
if __name__=='__main__':
    token_tag, set_token, set_tag, sentence_token_tag = preprocess(FILEPATH)
    ncv = NCrossValidation(sentence_token_tag, n=5, method='bigram')
    print('Bigram')
    ncv.run()
    ncv = NCrossValidation(sentence_token_tag, n=5, method='trigram')
    print('Trigram')
    ncv.run()
    ncv = NCrossValidation(sentence_token_tag, n=5, method='rnn')
    # print('RNN')
    # ncv.run()
            
            