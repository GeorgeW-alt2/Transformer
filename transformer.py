# Large Language Model v0.11 *Experimental*

import numpy as np
import random
import pickle

stop_words = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each",
    "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's",
    "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up",
    "very",
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
]

descale_factor = 0.1 # h_next_descale_factor
generate_len = 100
dictionary_memory_uncompressed = 1580 #Use -1 for large scale training
hidden_size = 1740 # adjust weights appropriately
epochs = 50 # no available metrics for suitable epoch count
compendium_filename = f"Compendium#{random.randint(0, 10000000)}.txt"
file = "test.txt"
# Define file names for saving and loading word_dict
word_dict_file = "word_dict.dat"

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases for the neural network
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        seq_len = inputs.shape[1]
        h_next = h_prev
        outputs = np.zeros((self.output_size, seq_len))

        for t in range(seq_len):
            x_t = inputs[:, t].reshape(-1, 1)
            h_next = np.tanh(np.dot(self.W_xh, x_t)) + np.dot(self.W_hh, h_prev) + self.b_h
            y_t = np.dot(self.W_hy, h_next) + self.b_y
            outputs[:, t] = y_t.squeeze()
            h_prev = h_next
        return outputs, h_next

def softmax(x):
    """Compute softmax function."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def preprocess_text(text, n=3):
    ngrams = []
    words = text.split()
    sentence_ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    for ngram in sentence_ngrams:
        ngram_str = ' '.join(ngram)
        ngrams.append(ngram_str)
    return list(ngrams)

def find_word_index(word_dict, input_word):
    results = []
    for n in range(len(word_dict)):
        try:
            input_index = word_dict[n].split().index(input_word)
            results.append(input_index)
        except:
            False
    return results

def generate_text_rnn(rnn, user_input, word_dict, length_to_generate):
    # Generate text using the RNN model
    generated_text = user_input
    h_prev = np.zeros((hidden_size, 1))
    input_vector = np.zeros((len(word_dict), 1))

    for word in ' '.join(generated_text).split():
        # Get the index of the input word in the list of unique words
        input_indexes = find_word_index(word_dict, word)
        for input_index in input_indexes:
            input_vector[input_index] = 1
    for _ in range(generate_len):
        # Forward pass
        output, h_prev = rnn.forward(input_vector, h_prev)
        # Apply temperature scaling to the output probabilities
        adjusted_probabilities = softmax(output.flatten())
        # Sample the next word index
        adjusted_probabilities[np.isnan(adjusted_probabilities)] = 0.1
        normalized_probabilities = adjusted_probabilities / np.sum(adjusted_probabilities)

        next_index = np.random.choice(len(normalized_probabilities), p=normalized_probabilities)

        # Get the next word from the unique_words list
        next_word = word_dict[next_index]
        if any(next_word.find(p) != -1 for p in stop_words):  # Check if next_word ends with punctuation
            input_vector = np.zeros((len(word_dict), 1))
        input_vector[next_index] = 1
        generated_text.append(next_word)

    return ' '.join(generated_text)

# Function to save word_dict to a file
def save_word_dict(word_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(word_dict, f)
    print(f"Word dictionary saved to {filename}")

# Function to load word_dict from a file
def load_word_dict(filename):
    with open(filename, 'rb') as f:
        word_dict = pickle.load(f)
    print(f"Word dictionary loaded from {filename}")
    return word_dict

def main():
    word_dict = None
    _choice_ = input("\nSave new model/Load old model?[s/l]:").lower()
    if (_choice_ == "l"):
        # Check if word_dict file exists and load if it does
        try:
            word_dict = load_word_dict(word_dict_file)
        except FileNotFoundError:
            word_dict = None

        # Define neural network parameters
        input_size = len(word_dict)
        output_size = len(word_dict)

        # Initialize RNN
        rnn = RNN(input_size, hidden_size, output_size)

    # If word_dict doesn't exist or if user chooses to generate a new one
    if word_dict is None or _choice_ == "s":
        with open(file, encoding="UTF-8") as f:
            text = f.read()
        print("Learning from:", file)
        # Get transition matrix and words list from the input text
        word_dict = preprocess_text(text)[:dictionary_memory_uncompressed]
        # Save word_dict to file
        save_word_dict(word_dict, word_dict_file)

        # Define neural network parameters
        input_size = len(word_dict)
        output_size = len(word_dict)

        # Initialize RNN
        rnn = RNN(input_size, hidden_size, output_size)

        # Initial hidden state (all zeros)
        h_prev = np.zeros((hidden_size, 1))

        # Train the RNN using the frequency map
    while True:
        u_input = input("USER: ").strip().lower().split()

        # Generate text using RNN
        rnn_generated_text = generate_text_rnn(rnn, u_input, word_dict, generate_len).lower()
        print("\nAI:", rnn_generated_text, "\n\n")

        # Write the answer to the file
        with open(compendium_filename, "a", encoding="utf8") as f:
            f.write(f"\nAnswering: {' '.join(u_input)}\n{rnn_generated_text}\n")

if __name__ == "__main__":
    main()
