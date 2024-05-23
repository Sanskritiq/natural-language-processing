from preprocess import data_preprocess, create_trigram_model, create_bigram_model
from model import HMMModel
import pandas as pd

def n_fold_testings(n, data):
    test_size = len(data)//n
    start_index = 0
    for i in range(n):
        data_test = data[start_index:start_index+test_size]
        data_train = data[:start_index]+data[start_index+test_size:]
        start_index += test_size
        yield (data_train, data_test)
        
def gram_input(word_tag):
    obs_seq = []
    bi_state_seq = []
    tri_state_seq = []
    for i in range(len(word_tag)):
        obs_seq.append(word_tag[i][0])
        bi_state_seq.append(word_tag[i][1])
        if i < len(word_tag)-1:
            tri_state_seq.append((word_tag[i][1], word_tag[i+1][1]))
        
        
    return obs_seq, bi_state_seq, tri_state_seq

def dataloader(data):
    for sentence_word_tag in data:
        yield gram_input(sentence_word_tag)
        

if __name__=="__main__":
    words, tags, word_tags, sentence_word_tags = data_preprocess()
    bigram_states, bigram_observations, bigram_start_prob, bigram_transition_prob, bigram_emission_prob = create_bigram_model(word_tags)
    trigram_states, trigram_observations, trigram_start_prob, trigram_transition_prob, trigram_emission_prob = create_trigram_model(word_tags)
    
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
    
    n = 5
    for train_data, test_data in n_fold_testings(n, sentence_word_tags):
        print('N-Fold Testing...\n')
        for test_seq, bi_state_seq, tri_state_seq in dataloader(test_data):
            
            print('Testing...\n')
            bigram_predict = bigram_hmm_model.predict(test_seq)
            trigram_predict = trigram_hmm_model.predict(test_seq)
            bigram_predict = list(map(lambda x: bigram_hmm_model.states[x], bigram_predict))
            trigram_predict = list(map(lambda x: trigram_hmm_model.states[x], trigram_predict))
            print(f'Bigram HMM Model Predicted Sequence: {bigram_predict}')
            print(f'Trigram HMM Model Predicted Sequence: {trigram_predict}')

            print(f'Bigram HMM Model Accuracy: {bigram_hmm_model.accuracy(test_seq, bi_state_seq):.2%}')
            print(f'Trigram HMM Model Accuracy: {trigram_hmm_model.accuracy(test_seq, tri_state_seq):.2%}')
            
            print('Testing done!\n')
        print('N-Fold Testing done!\n')
        