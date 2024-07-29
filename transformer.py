# Large Language Model v17.7 X

import numpy as np
import pickle
import re
from concurrent.futures import ThreadPoolExecutor

# Model parameters
KB_memory_uncompressed = -1  # KB access, -1 for unlimited
generate_length = 100
n = 3
sigma = 0.7  # Width of the Gaussian functions

padding_token = '<unk>'

def create_ngrams_and_words(text, max_n):
    words = text.split()
    ngrams_and_words = words.copy()  # Start with single words
    for n in range(2, max_n + 1):
        ngrams = zip(*[words[i:] for i in range(n)])
        ngrams_and_words.extend([' '.join(ngram) for ngram in ngrams])
    return ngrams_and_words

def gaussian_rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - np.dot(x, np.dot(x, c)))**2 / (2 * s**2))

def encode_ngram(ngram, token_vector, word_to_idx, centers, sigma):
    if ngram in word_to_idx:
        idx = word_to_idx[ngram]
        return idx, gaussian_rbf(token_vector, centers[idx], sigma)
    else:
        idx = word_to_idx[padding_token]
        return idx, gaussian_rbf(token_vector, centers[idx], sigma)

def encode_sentence(sentence, word_to_idx, centers, sigma, max_n):
    encoded = np.zeros(len(word_to_idx))
    tokens = create_ngrams_and_words(sentence, max_n)

    token_vector = np.zeros(len(word_to_idx))
    for token in tokens:
        if token in word_to_idx:
            token_vector[word_to_idx[token]] = 1
        else:
            token_vector[word_to_idx[padding_token]] = 1

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(encode_ngram, ngram, token_vector, word_to_idx, centers, sigma) for ngram in tokens]
        for future in futures:
            idx, rbf_value = future.result()
            encoded[idx] = rbf_value

    return encoded

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps)

def chat(ngram_encoding_index, question, word_to_idx, generate_length, n):
    output = []
    encoded = np.zeros(len(word_to_idx))

    ngrams = create_ngrams_and_words(question, n)
    for ngram in ngrams:
        if ngram in ngram_encoding_index:
            idx, rbf_value = ngram_encoding_index[ngram]
            encoded[idx] = rbf_value
    input_seq = encoded

    for i in range(generate_length):
        probabilities = softmax(input_seq.flatten())
        rng = np.random.default_rng()
        predicted_idx = rng.choice(range(len(probabilities)), p=probabilities)
        ngram = idx_to_word.get(predicted_idx, padding_token)
        output.append(ngram)

        encoded = np.zeros(len(word_to_idx))
        next_input = ' '.join(output)
        ngrams = create_ngrams_and_words(next_input, n)
        for ngram in ngrams:
            if ngram in ngram_encoding_index:
                idx, rbf_value = ngram_encoding_index[ngram]
                encoded[idx] = rbf_value
        input_seq = encoded

    generated_response = ' '.join(output)
    return generated_response

# Function to save a dictionary to a file
def save_dict(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f"Dictionary saved to {filename}")

# Function to load a dictionary from a file
def load_dict(filename):
    with open(filename, 'rb') as f:
        dictionary = pickle.load(f)
    print(f"Dictionary loaded from {filename}")
    return dictionary

def remove_sentences_with_numbers_and_symbols(sentences):
    filtered_sentences = []
    for sentence in sentences:
        if re.match(r'^[A-Za-z\s,.]+$', sentence):
            filtered_sentences.append(sentence)
    return filtered_sentences

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

_choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

word_to_idx = {}
idx_to_word = {}
ngram_encoding_index = {}
if _choice_ == "s":
    # Load and preprocess data
    with open("test.txt", encoding="UTF-8") as f:
        conversations = f.read().lower().split(".")[:KB_memory_uncompressed]
    conversations = remove_sentences_with_numbers_and_symbols(conversations)
    print("Memory size: ", len(conversations))
    
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
    save_dict(word_to_idx, "langA.dat")
    save_dict(idx_to_word, "langB.dat")

    # Create n-gram encoding index
    centers = np.linspace(-1, 1, len(word_to_idx))  # Identity matrix as a simple example for centers
    total_ngrams = len(vocab)
    current_progress = 0
    print_progress_bar(current_progress, total_ngrams, prefix='Progress:', suffix='Complete', length=50)
    for ngram in vocab:
        token_vector = np.zeros(len(word_to_idx))
        if ngram in word_to_idx:
            token_vector[word_to_idx[ngram]] = 1
        else:
            token_vector[word_to_idx[padding_token]] = 1
        idx, rbf_value = encode_ngram(ngram, token_vector, word_to_idx, centers, sigma)
        ngram_encoding_index[ngram] = (idx, rbf_value)
        current_progress += 1
        print_progress_bar(current_progress, total_ngrams, prefix='Progress:', suffix='Complete', length=50)
    
    save_dict(ngram_encoding_index, "model.dat")

if _choice_ == "l":
    word_to_idx = load_dict("langA.dat")
    idx_to_word = load_dict("langB.dat")
    ngram_encoding_index = load_dict("model.dat")

# Example usage
while True:
    user_input = input("You: ")
    response = chat(ngram_encoding_index, user_input, word_to_idx, generate_length, n)
    print(f"AI: {response}")
