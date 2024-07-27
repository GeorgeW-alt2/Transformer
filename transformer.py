# Large Language Model v14.1
import numpy as np
import pickle
import re

# Model parameters
KB_memory_uncompressed = -1  # KB access, -1 for unlimited
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
    reuptake = np.zeros(len(word_to_idx))

    for ngram in tokens:
        if ngram in word_to_idx:
            idx = word_to_idx[ngram]
            encoded[idx] = 2
            reuptake[idx+1] = -1
        else:
            encoded[word_to_idx[padding_token]] = 1
    encoded += reuptake
    return encoded

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps)

def chat(question, word_to_idx, generate_length, n):
    output = []
    input_seq = encode_sentence(question, word_to_idx, n)
    
    for i in range(generate_length):
        probabilities = softmax(input_seq.flatten())
        rng = np.random.default_rng()
        predicted_idx = rng.choice(range(len(probabilities)), p=probabilities)
        ngram = idx_to_word.get(predicted_idx, padding_token)
        output.append(ngram)

        next_input = ' '.join(output)
        input_seq = encode_sentence(next_input, word_to_idx, n)

    generated_response = ' '.join(output)
    return generated_response

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

# Example usage
while True:
    user_input = input("You: ")
    response = chat(user_input, word_to_idx, generate_length, n)
    print(f"AI: {response}")
