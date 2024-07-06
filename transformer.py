# Large Language Model v0.16 *Experimental*

import numpy as np
import random
import pickle

generate_len = 100
dictionary_memory_uncompressed = 580  # Use -1 for large scale training
hidden_size = 740  # adjust weights appropriately
epochs = 50  # no available metrics for suitable epoch count
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

    def forward(self, inputs, h_prev, word_dict):
        seq_len = inputs.shape[1]
        h_next = h_prev
        outputs = np.zeros((self.output_size, seq_len))

        for t in range(seq_len):
            x_t = inputs[:, t].reshape(-1, 1)
            h_next = np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
            y_t = np.dot(self.W_hy, h_next) + self.b_y
            outputs[:, t] = y_t.squeeze()
            h_prev = h_next

            # Reorder word_dict based on the output probabilities
            sorted_indices = np.argsort(outputs[:, t])[::-1]  # Sort in descending order
            word_dict = [word_dict[i] for i in sorted_indices]

        return outputs, h_next, word_dict


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
        ngrams.append(ngram_str.split()[1])
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
    generated_text = user_input[:]
    h_prev = np.zeros((hidden_size, 1))
    input_vector = np.zeros((len(word_dict), 1))

    for word in generated_text:
        input_indexes = find_word_index(word_dict, word)
        for input_index in input_indexes:
            input_vector[input_index] = 1

    for _ in range(length_to_generate):
        # Forward pass
        output, h_prev, word_dict = rnn.forward(input_vector, h_prev, word_dict)
        adjusted_probabilities = softmax(output.flatten())

        rng = np.random.default_rng()
        next_index = np.random.choice(rng.permutation(len(adjusted_probabilities)), p=adjusted_probabilities)

        # Get the next word from the reordered word_dict
        next_word = word_dict[next_index]
        generated_text.append(next_word)

        # Prepare the input vector for the next step
        input_vector = np.zeros((len(word_dict), 1))
        input_vector[next_index] = 1

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
