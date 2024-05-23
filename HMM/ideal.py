import nltk
from nltk import word_tokenize
from nltk.corpus import brown
from nltk.tag import hmm

# Step 1: Load and preprocess the Brown corpus
brown_sents = brown.tagged_sents()

# Step 2: Split the dataset into training and testing sets
train_size = int(0.8 * len(brown_sents))
train_data = brown_sents[:train_size]
test_data = brown_sents[train_size:]

# Step 3: Train the HMM models (bigram and trigram)
trainer_bigram = hmm.HiddenMarkovModelTrainer(tag_set=None, backoff=hmm.HiddenMarkovModelTrainer.BigramTagger)
trainer_trigram = hmm.HiddenMarkovModelTrainer(tag_set=None, backoff=hmm.HiddenMarkovModelTrainer.TrigramTagger)

hmm_model_bigram = trainer_bigram.train(train_data)
hmm_model_trigram = trainer_trigram.train(train_data)

# Step 4: Evaluate the models on the test set
accuracy_bigram = hmm_model_bigram.evaluate(test_data)
accuracy_trigram = hmm_model_trigram.evaluate(test_data)

print(f"Accuracy (Bigram): {accuracy_bigram:.2%}")
print(f"Accuracy (Trigram): {accuracy_trigram:.2%}")

# Example of tagging a new sentence
new_sentence = "The cat sat on the mat."
tokenized_sentence = word_tokenize(new_sentence.lower())  # Tokenize and convert to lowercase
tagged_bigram = hmm_model_bigram.tag(tokenized_sentence)
tagged_trigram = hmm_model_trigram.tag(tokenized_sentence)

print(f"Tagged (Bigram): {tagged_bigram}")
print(f"Tagged (Trigram): {tagged_trigram}")
