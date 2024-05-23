import numpy as np
import pandas as pd
from preprocess import data_preprocess, create_bigram_model, create_trigram_model

  
class HMMModel:
    def __init__(self, states, observations, start_prob, transition_prob, emission_prob):
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        
    
    def viterbi(self, obs_seq):
        delta = np.zeros((len(obs_seq), len(self.states)))
        psi = np.zeros((len(obs_seq), len(self.states)))
        for i, obs in enumerate(obs_seq):
            for j, state in enumerate(self.states):
                if i == 0:
                    delta[i][j] = self.start_prob[state] * self.emission_prob[state][obs]
                else:
                    delta[i][j] = max([delta[i-1][k] * self.transition_prob[state2][state] for k, state2 in enumerate(self.states)]) * self.emission_prob[state][obs]
                    psi[i][j] = np.argmax([delta[i-1][k] * self.transition_prob[state2][state] for k, state2 in enumerate(self.states)])
        return delta, psi
    
    def predict(self, obs_seq):
        delta, psi = self.viterbi(obs_seq)
        path = [np.argmax(delta[-1])]
        for i in range(len(obs_seq)-1, 0, -1):
            path.insert(0, int(psi[i][path[0]]))
        return path
    
    def accuracy(self, obs_seq, state_seq):
        pred_state_seq = self.predict(obs_seq)
        return sum([1 for i in range(len(state_seq)) if state_seq[i] == self.states[pred_state_seq[i]]]) / len(state_seq)
    
    
    
if __name__=="__main__":
    words, tags, word_tags, sentence_word_tags = data_preprocess()
    bigram_states, bigram_observations, bigram_start_prob, bigram_transition_prob, bigram_emission_prob = create_bigram_model(word_tags)
    trigram_states, trigram_observations, trigram_start_prob, trigram_transition_prob, trigram_emission_prob = create_trigram_model(word_tags)
    # START       ADP       DET      NOUN      VERB       ADJ      CONJ       PRT         .  END       ADV       NUM      PRON         X
    BI_TEST_SEQ = ['<start>', 'i', 'am', 'contributing', 'in', 'the', 'growth', '<end>']
    BI_STATE_SEQ = ['START', 'PRON', 'VERB', 'VERB', 'ADP', 'DET', 'NOUN', 'END']
    
    TRI_TEST_SEQ = [('<start>', 'i'), ('i', 'am'), ('am', 'contributing'), ('contributing', 'in'), ('in', 'the'), ('the', 'growth'), ('growth', '<end>')]
    TRI_STATE_SEQ = [('START', 'PRON'), ('PRON', 'VERB'), ('VERB', 'VERB'), ('VERB', 'ADP'), ('ADP', 'DET'), ('DET', 'NOUN'), ('NOUN', 'END')]
    
    bigram_hmm_model = HMMModel(bigram_states, bigram_observations, bigram_start_prob, bigram_transition_prob, bigram_emission_prob)
    trigram_hmm_model = HMMModel(trigram_states, trigram_observations, trigram_start_prob, trigram_transition_prob, trigram_emission_prob)
    
    bigram_transition_prob_new, bigram_emission_prob_new = pd.DataFrame(bigram_transition_prob), pd.DataFrame(bigram_emission_prob)
    trigram_transition_prob_new, trigram_emission_prob_new = pd.DataFrame(trigram_transition_prob), pd.DataFrame(trigram_emission_prob)
    print('Bigram start probability')
    print(bigram_start_prob)
    print('Bigram transition probability')
    print(bigram_transition_prob_new.head())
    print('Bigram emission probability')
    print(bigram_emission_prob_new.head())
    print('Trigram start probability')
    print(trigram_start_prob)
    print('Trigram transition probability')
    print(trigram_transition_prob_new.head())
    print('Trigram emission probability')
    print(trigram_emission_prob_new.head())
    
    print(f"Bigram HMM Model Accuracy: {bigram_hmm_model.evaluate(BI_TEST_SEQ):.2%}")
    print(f"Trigram HMM Model Accuracy: {trigram_hmm_model.evaluate(TRI_TEST_SEQ):.2%}")
    
    print(f'Bigram HMM Model Predicted Sequence: {bigram_hmm_model.predict(BI_TEST_SEQ)}')
    print(f'Trigram HMM Model Predicted Sequence: {trigram_hmm_model.predict(TRI_TEST_SEQ)}')
    
    print(f'Bigram HMM Model Accuracy: {bigram_hmm_model.accuracy(BI_TEST_SEQ, BI_STATE_SEQ):.2%}')
    print(f'Trigram HMM Model Accuracy: {trigram_hmm_model.accuracy(TRI_TEST_SEQ, TRI_STATE_SEQ):.2%}')