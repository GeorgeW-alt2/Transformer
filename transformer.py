# Large Language Model v5.7 - George W

import numpy as np
import pickle
import re
import math

# Model parameters
KB_memory_uncompressed = 100  # KB access, -1 for unlimited
generate_length = 100
epochs = 5
n = 3
D = 200  # Dimensionality of the RFF mapping
learning_rate = 0.01  # Learning rate for training
padding_token = '<unk>'
# Create n-grams and filter out n-grams with symbols
def create_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

# Check if an n-gram contains only alphanumeric characters and spaces
def is_valid_ngram(ngram):
    return re.match("^[a-zA-Z0-9\s.]*$", ngram) is not None

# Encoding function with <unk> token handling
def encode_sentence(sentence, word_to_idx, n):
    encoded = np.zeros(len(word_to_idx))
    ngrams = create_ngrams(sentence, n)
    for ngram in ngrams:
        if ngram in word_to_idx:
            encoded[word_to_idx[ngram] - 1] = 1
        else:
            encoded[word_to_idx[padding_token] - 1] = 0
    return encoded

# Random Fourier Features transformation
def rff_mapping(input_vec, W, b):
    feature_vec = []
    for i in range(D):
        random_projection = max(W[i][j] * input_vec[j] for j in range(len(input_vec)))
        feature = math.sqrt(2/D) * math.tanh(random_projection + b[i])
        feature_vec.append(feature)
    return feature_vec

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps)

def chat(question, word_to_idx, generate_length, n, W, b):
    output = []
    input_seq = encode_sentence(question, word_to_idx, n)
    rff_input = np.array(rff_mapping(input_seq, W, b))

    for i in range(generate_length):
        adjusted_probabilities = softmax(rff_input.flatten())

        rng = np.random.default_rng()
        predicted_idx = rng.choice(range(len(adjusted_probabilities)), p=adjusted_probabilities)
        if predicted_idx + 1 in idx_to_word:  # Adjust index to start from 0
            output.append(idx_to_word[predicted_idx + 1])
        else:
            output.append(padding_token)

        next_input = ' '.join(output)
        input_seq = encode_sentence(' '.join(output), word_to_idx, n)
        rff_input = np.array(rff_mapping(input_seq, W, b))

    return ' '.join(output)

# Function to save word_dict from a file
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

# Function to save RFF parameters to a file
def save_rff_params(W, b, filename):
    with open(filename, 'wb') as f:
        pickle.dump((W, b), f)
    print(f"RFF parameters saved to {filename}")

# Function to load RFF parameters from a file
def load_rff_params(filename):
    with open(filename, 'rb') as f:
        W, b = pickle.load(f)
    print(f"RFF parameters loaded from {filename}")
    return W, b

def train_rff(sentences, word_to_idx, n, W, b, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for sentence in sentences:
            input_seq = encode_sentence(sentence, word_to_idx, n)
            rff_input = np.array(rff_mapping(input_seq, W, b))
            target_seq = encode_sentence(sentence, word_to_idx, n)

            # Ensure target_seq has the same shape as rff_input
            if target_seq.shape != rff_input.shape:
                target_seq = target_seq[:rff_input.shape[0]]

            # Roll the target sequence
            target_seq = np.roll(target_seq, shift=-1)

            # Calculate error
            error = rff_input - target_seq

            # Update weights and biases
            W -= learning_rate * np.outer(error, input_seq)
            b -= learning_rate * error

            # Calculate loss (Mean Squared Error)
            loss = np.mean(error ** 2)
            total_loss += loss

        avg_loss = total_loss / len(sentences)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

_choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

word_to_idx = {}
idx_to_word = {}
if _choice_ == "s":
    # Load and preprocess data
    with open("test.txt", encoding="UTF-8") as f:
        conversations = f.read().lower().split(".")[:KB_memory_uncompressed]

    # Vocabulary creation including PMI values
    vocab = set()
    for conv in conversations:
        ngrams = create_ngrams(conv, n)
        for ngram in ngrams:
            if is_valid_ngram(ngram):
                vocab.add(ngram)

    # Add a special token for unknown words
    vocab.add(padding_token)

    # Process word dictionary
    word_to_idx = {word: idx for idx, word in enumerate(vocab, start=0)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    save_word_dict(word_to_idx, "langA.dat")
    save_word_dict(idx_to_word, "langB.dat")

    # Initialize RFF parameters
    W = np.random.randn(D, len(word_to_idx)) * 0.01
    b = np.random.uniform(0, 2 * np.pi, D)

    # Train the model
    train_rff(conversations, word_to_idx, n, W, b, learning_rate, epochs)
    save_rff_params(W, b, "rff_params.dat")  # Save trained parameters

if _choice_ == "l":
    word_to_idx = load_word_dict("langA.dat")
    idx_to_word = load_word_dict("langB.dat")
    W, b = load_rff_params("rff_params.dat")

# Example usage
while True:
    user_input = input("You: ")
    response = chat(user_input, word_to_idx, generate_length, n, W, b)
    print(f"AI: {response}")
