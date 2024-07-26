# Large Language Model v11.8

import numpy as np
import pickle
import re

# Model parameters
KB_memory_uncompressed = 1000 # KB access, -1 for unlimited
generate_length = 25
n = 3
padding_token = '<unk>'

def create_ngrams_and_words(text, max_n):
    words = text.split()
    ngrams_and_words = words.copy()  # Start with single words
    for n in range(2, max_n + 1):
        ngrams = zip(*[words[i:] for i in range(n)])
        ngrams_and_words.extend([' '.join(ngram) for ngram in ngrams])
    return ngrams_and_words

def encode_sentence(sentence, word_to_idx, max_n):
    encoded = np.zeros(len(word_to_idx))
    tokens = create_ngrams_and_words(sentence, max_n)
    for ngram in tokens:
        probabilities = softmax(encoded.flatten())
        if ngram in word_to_idx:
            encoded[word_to_idx[ngram]] = probabilities[-1]
        else:
            encoded[word_to_idx[padding_token]] = 1
    return encoded

def softmax(logits):
    exps = np.exp(logits - np.max(logits)*-1)  # Subtract max for numerical stability
    return exps / np.sum(exps)
    
def nn_forward(X, W1, b1, W2, b2):
     return np.exp(np.maximum(0, np.dot(np.dot(X, W1) + b1, W2) + b2)) / np.sum(np.exp(np.maximum(0, np.dot(np.dot(X, W1) + b1, W2) + b2)), axis=-1, keepdims=True)

def chat(question, word_to_idx, generate_length, n):
    output = []
    input_seq = encode_sentence(question, word_to_idx, n)
    
    for i in range(generate_length):
        input_seq_reshaped = softmax(input_seq.reshape(1, -1))  # Batch size of 1
        probabilities = nn_forward(input_seq_reshaped, W1, b1, W2, b2).flatten()

        rng = np.random.default_rng()
        predicted_idxs = rng.choice(range(len(probabilities)), p=probabilities, size=2)
        for idx in predicted_idxs:
            ngram = idx_to_word.get(idx, padding_token)
            output.append(ngram)

        next_input = ' '.join(output)
        input_seq = encode_sentence(' '.join(output), word_to_idx, n)
    
    return ' '.join(output).strip()

# Function to save word_dict to a file
def save_word_dict(word_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(word_dict, f)
    print(f"Dictionary saved to {filename}")

# Function to load word_dict from a file
def load_word_dict(filename):
    with open(filename, 'rb') as f:
        word_dict = pickle.load(f)
    print(f"Dictionary loaded from {filename}")
    return word_dict

_choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

word_to_idx = {}
idx_to_word = {}
if _choice_ == "s":
    # Load and preprocess data
    with open("test.txt", encoding="UTF-8") as f:
        conversations = f.read().lower().split(".")[:KB_memory_uncompressed]

    # Vocabulary creation
    vocab = set()
    for conv in conversations:
        ngrams = create_ngrams_and_words(conv + ".", n)
        for ngram in ngrams:
            vocab.add(ngram)

    # Add a special token for unknown words
    vocab.add(padding_token)

    # Process word dictionary
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}  # Start indexing from 1
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    save_word_dict(word_to_idx, "langA.dat")
    save_word_dict(idx_to_word, "langB.dat")

if _choice_ == "l":
    word_to_idx = load_word_dict("langA.dat")
    idx_to_word = load_word_dict("langB.dat")

# Initialize weights and biases
input_size = len(word_to_idx)
hidden_size = 128
output_size = len(word_to_idx)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

# Example usage
while True:
    user_input = input("You: ")
    response = chat(user_input, word_to_idx, generate_length, n)
    print(f"AI: {response}")
