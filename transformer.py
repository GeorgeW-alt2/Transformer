
# Large Language Model v4.1 *Experimental*
import numpy as np
import pickle

# Model parameters
KB_memory_uncompressed = -1 # KB access, -1 for unlimited
generate_length = 100
padding_token = '<unk>'
n = 3

# Create n-grams
def create_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

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

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.exp(exps)

def chat(question, word_to_idx, generate_length, n):
    output = []
    input_seq = encode_sentence(question, word_to_idx, n*3)

    for i in range(generate_length):
        adjusted_probabilities = softmax(input_seq.flatten())

        # Invert the adjusted probabilities
        inverted_probabilities = 1 / adjusted_probabilities
        inverted_probabilities /= inverted_probabilities.sum()  # Normalize to ensure they sum to 1

        rng = np.random.default_rng()
        predicted_idx = rng.choice(range(len(inverted_probabilities)), p=inverted_probabilities)
        if predicted_idx + 1 in idx_to_word:  # Adjust index to start from 0
            output.append(idx_to_word[predicted_idx + 1])
        else:
            output.append(padding_token)

        input_seq = encode_sentence(' '.join(output), word_to_idx, n)[:, np.newaxis].T

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

_choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

word_to_idx = {}
idx_to_word = {}
if (_choice_ == "s"):
    # Load and preprocess data
    with open("test.txt", encoding="UTF-8") as f:
        conversations = f.read().lower().split(".")[:KB_memory_uncompressed]

    # Vocabulary creation including PMI values
    vocab = set()
    for conv in conversations:
        ngrams = create_ngrams(conv, n)
        for ngram in ngrams:
            vocab.add(ngram)

    # Add a special token for unknown words
    vocab.add(padding_token)

    # Process word dictionary
    word_to_idx = {word: idx for idx, word in enumerate(vocab, start=1)}  # Start indexing from 1
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    save_word_dict(word_to_idx, "langA.dat")
    save_word_dict(idx_to_word, "langB.dat")

if (_choice_ == "l"):
    word_to_idx = load_word_dict("langA.dat")
    idx_to_word = load_word_dict("langB.dat")

# Example usage
while True:
    user_input = input("You: ")
    response = chat(user_input, word_to_idx, generate_length, n)
    print(f"AI: {response}")
