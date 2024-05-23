import numpy as  np
import pandas as  pd
import matplotlib.pyplot as plt
import nltk

DATA_PATH = 'Brown_train.txt'

def data_preprocess(data_path: str = DATA_PATH):
    print('Preprocessing data...')
    words = []
    tags = []
    word_tags = []
    sentence_word_tags = []
    for line in open(data_path):
        word_tag= line.strip().split()
        sentence_wt = []
        is_start = True
        for word_tag in line.strip().split():
            try:
                word, tag = word_tag.split('/')
            except:
                tag = word_tag.split('/')[-1]
                word = '/'.join(word_tag.split('/')[:-1]).strip()
                # print(word_tag, word, tag)
            # if for_testing:
            #     testing_data.append((word, tag))
            #     sentence = ' '.join([x[0] for x in testing_data])
            if is_start:
                is_start = False
                word_tags.append(('<start>', 'START'))
            word = word.lower()
            if word not in words:
                words.append(word)
            if tag not in tags:
                tags.append(tag)
            word_tags.append((word, tag))
            sentence_wt.append((word, tag))
        sentence_word_tags.append(sentence_wt)
        word_tags.append(('<end>', 'END'))
    words.append('<start>')
    words.append('<end>')
    print('Data preprocessing done!')
    return words, tags, word_tags, sentence_word_tags

# create bigram states and observations
def create_bigram_states_observations(word_tags):
    print('Creating Bigram states and observations...')
    states = []
    observations = []
    for word, tag in word_tags:
        if tag not in states:
            states.append(tag)
        if word not in observations:
            observations.append(word)
    print('Bigram states and observations created!')
    return states, observations

# create bigram start probability
def create_bigram_start_prob(states, word_tags):
    print('Creating Bigram start probability...')
    start_prob = {}
    for state in states:
        start_prob[state] = 0
    for word, tag in word_tags:
        if word == '<start>':
            start_prob[tag] += 1
    for state in states:
        start_prob[state] = start_prob[state] / len(word_tags)
    print('Bigram start probability created!')
    return start_prob

# create bigram transition probability
def create_bigram_transition_prob(states, word_tags):
    print('Creating Bigram transition probability...')
    transition_prob = {}
    for state in states:
        transition_prob[state] = {}
        for state2 in states:
            transition_prob[state][state2] = 0
    for i in range(len(word_tags)-1):
        if word_tags[i][0] == '<end>':
            continue
        transition_prob[word_tags[i][1]][word_tags[i+1][1]] += 1
    for state in states:
        total = sum([transition_prob[state][state2] for state2 in states])
        for state2 in states:
            try:
                transition_prob[state][state2] = transition_prob[state][state2] / total
            except ZeroDivisionError:
                transition_prob[state][state2] = 0
    print('Bigram transition probability created!')
    return transition_prob

# create bigram emission probability
def create_bigram_emission_prob(states, observations, word_tags):
    print('Creating Bigram emission probability...')
    emission_prob = {}
    for state in states:
        emission_prob[state] = {}
        for observation in observations:
            emission_prob[state][observation] = 0
    for word, tag in word_tags:
        emission_prob[tag][word] += 1
    for state in states:
        total = sum([emission_prob[state][observation] for observation in observations])
        for observation in observations:
            try:
                emission_prob[state][observation] = emission_prob[state][observation] / total
            except ZeroDivisionError:
                emission_prob[state][observation] = 0
    print('Bigram emission probability created!')
    return emission_prob

# create bigram model
def create_bigram_model(word_tags):
    print('Creating Bigram model...')
    states, observations = create_bigram_states_observations(word_tags)
    start_prob = create_bigram_start_prob(states, word_tags)
    transition_prob = create_bigram_transition_prob(states, word_tags)
    emission_prob = create_bigram_emission_prob(states, observations, word_tags)
    print('Bigram model created!')
    return states, observations, start_prob, transition_prob, emission_prob

# create trigram states and observations
def create_trigram_states_observations(word_tags):
    print('Creating Trigram states and observations...')
    states = []
    observations = []
    for i in range(len(word_tags)-1):
        if word_tags[i][0] == '<end>':
            continue
        if (word_tags[i][1], word_tags[i+1][1]) not in states:
            states.append((word_tags[i][1], word_tags[i+1][1]))
        if word_tags[i][0] not in observations:
            observations.append(word_tags[i][0])
    print('Trigram states and observations created!')
    return states, observations

# create trigram start probability
def create_trigram_start_prob(states, word_tags):
    print('Creating Trigram start probability...')
    start_prob = {}
    for state in states:
        start_prob[state] = 0
    for i in range(len(word_tags)-1):
        if word_tags[i][0] == '<start>':
            start_prob[(word_tags[i][1], word_tags[i+1][1])] += 1
    for state in states:
        start_prob[state] = start_prob[state] / len(word_tags)
    print('Trigram start probability created!')
    return start_prob

# create trigram transition probability
def create_trigram_transition_prob(states, word_tags):
    print('Creating Trigram transition probability...')
    transition_prob = {}
    for state in states:
        transition_prob[state] = {}
        for state2 in states:
            transition_prob[state][state2] = 0
    for i in range(len(word_tags)-3):
        if word_tags[i][0] == '<end>' or word_tags[i+1][0] == '<end>' or word_tags[i+2][0] == '<end>':
            continue
        transition_prob[(word_tags[i][1], word_tags[i+1][1])][(word_tags[i+2][1], word_tags[i+3][1])] += 1
    for state in states:
        total = sum([transition_prob[state][state2] for state2 in states])
        for state2 in states:
            try:
                transition_prob[state][state2] = transition_prob[state][state2] / total
            except ZeroDivisionError:
                transition_prob[state][state2] = 0
    print('Trigram transition probability created!')
    return transition_prob

# create trigram emission probability
def create_trigram_emission_prob(states, observations, word_tags):
    print('Creating Trigram emission probability...')
    emission_prob = {}
    for state in states:
        emission_prob[state] = {}
        for observation in observations:
            emission_prob[state][observation] = 0
    for i in range(len(word_tags)-1):
        if word_tags[i][0] == '<end>':
            continue
        emission_prob[(word_tags[i][1], word_tags[i+1][1])][word_tags[i][0]] += 1
    for state in states:
        total = sum([emission_prob[state][observation] for observation in observations])
        for observation in observations:
            try:
                emission_prob[state][observation] = emission_prob[state][observation] / total
            except ZeroDivisionError:
                emission_prob[state][observation] = 0
    print('Trigram emission probability created!')
    return emission_prob

# create trigram model
def create_trigram_model(word_tags):
    print('Creating Trigram model...')
    states, observations = create_trigram_states_observations(word_tags)
    start_prob = create_trigram_start_prob(states, word_tags)
    transition_prob = create_trigram_transition_prob(states, word_tags)
    emission_prob = create_trigram_emission_prob(states, observations, word_tags)
    print('Trigram model created!')
    return states, observations, start_prob, transition_prob, emission_prob

def create_states_observations(sentence_word_tags):
    print('Creating states and observations...')
    total_bigram_states, total_bigram_observations, total_trigram_states, total_trigram_observations = set(), set(), set(), set()
    for word_tag_pair in sentence_word_tags:
        bigram_states, bigram_observations = create_bigram_states_observations(word_tag_pair)
        trigram_states, trigram_observations = create_trigram_states_observations(word_tag_pair)
        total_bigram_states = total_bigram_states.union(set(bigram_states))
        total_bigram_observations = total_bigram_observations.union(set(bigram_observations))
        total_trigram_states = total_trigram_states.union(set(trigram_states))
        total_bigram_observations = total_trigram_observations.union(set(trigram_observations))
        # print(total_bigram_observations)
        
    total_bigram_states = list(total_bigram_states)
    total_bigram_observations = list(total_bigram_observations)
    total_trigram_states = list(total_trigram_states)
    total_trigram_observations = list(total_trigram_observations)
    print('States and observations created!')    
        
    return total_bigram_states, total_bigram_observations, total_trigram_states, total_trigram_observations



    
if __name__ == '__main__':
    words, tags, word_tags, sentence_word_tags = data_preprocess(DATA_PATH)
    print(words[:10])
    print(tags)
    print(word_tags[:100])
    print(len(words), len(tags), len(word_tags), len(sentence_word_tags))
    bigram_states, bigram_observations, bigram_start_prob, bigram_transition_prob, bigram_emission_prob = create_bigram_model(word_tags)
    trigram_states, trigram_observations, trigram_start_prob, trigram_transition_prob, trigram_emission_prob = create_trigram_model(word_tags)
    bigram_start_prob, bigram_transition_prob, bigram_emission_prob = pd.DataFrame(bigram_start_prob), pd.DataFrame(bigram_transition_prob), pd.DataFrame(bigram_emission_prob)
    trigram_start_prob, trigram_transition_prob, trigram_emission_prob = pd.DataFrame(trigram_start_prob), pd.DataFrame(trigram_transition_prob), pd.DataFrame(trigram_emission_prob)
    print('Bigram start probability')
    print(bigram_start_prob.head())
    print('Bigram transition probability')
    print(bigram_transition_prob.head())
    print('Bigram emission probability')
    print(bigram_emission_prob.head())
    print('Trigram start probability')
    print(trigram_start_prob.head())
    print('Trigram transition probability')
    print(trigram_transition_prob.head())
    print('Trigram emission probability')
    print(trigram_emission_prob.head())
    print('done')