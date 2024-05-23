import numpy as np
import pandas as pd
import torch

class HMMModel():
    def __init__(self, states, observations, transition_probabilities, emission_probabilities, start_probabilities):
        self.states = states
        self.observations = observations
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        self.start_probabilities = start_probabilities
        
    def viterbi(self, observations):
        T = len(observations)
        N = len(self.states)
        viterbi = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        
        # initialization
        for s in range(N):
            viterbi[s, 0] = self.start_probabilities[s] * self.emission_probabilities[s, observations[0]]
            backpointer[s, 0] = 0
            
        # recursion
        for t in range(1, T):
            for s in range(N):
                viterbi[s, t] = np.max([viterbi[s_, t-1] * self.transition_probabilities[s_, s] * self.emission_probabilities[s, observations[t]] for s_ in range(N)])
                backpointer[s, t] = np.argmax([viterbi[s_, t-1] * self.transition_probabilities[s_, s] for s_ in range(N)])
                
        # termination
        bestpathprob = np.max([viterbi[s, T-1] for s in range(N)])
        bestpathpointer = np.argmax([viterbi[s, T-1] for s in range(N)])
        
        # backtracking
        bestpath = [bestpathpointer]
        for t in range(T-1, 0, -1):
            bestpathpointer = backpointer[bestpathpointer, t]
            bestpath.insert(0, bestpathpointer)
            
        return bestpath, bestpathprob
    
    def viterbi_new(self, observations):
        T = len(observations)
        N = len(self.states)
        viterbi = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        
        # initialization
        for s in range(N):
            viterbi[s, 0] = self.start_probabilities[self.states[s]] * self.emission_probabilities[self.states[s]][observations[0]]
            backpointer[s, 0] = 0
            
        # recursion
        for t in range(1, T):
            for s in range(N):
                viterbi[s, t] = np.max([viterbi[s_, t-1] * self.transition_probabilities[self.states[s_]][self.states[s]] * self.emission_probabilities[self.states[s]][observations[t]] for s_ in range(N)])
                backpointer[s, t] = np.argmax([viterbi[s_, t-1] * self.transition_probabilities[self.states[s_]][self.states[s]] for s_ in range(N)])
                
        # termination
        bestpathprob = np.max([viterbi[s, T-1] for s in range(N)])
        bestpathpointer = np.argmax([viterbi[s, T-1] for s in range(N)])
        
        # backtracking
        bestpath = [bestpathpointer]
        for t in range(T-1, 0, -1):
            bestpathpointer = backpointer[bestpathpointer, t]
            bestpath.insert(0, bestpathpointer)
        
        # bestpath_list = []
        # for s, t in bestpath:
        #     bestpath_list.append(s)    
            
            
        return bestpath, bestpathprob
    
# rnn model for NER

class RNNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(input_size + hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    