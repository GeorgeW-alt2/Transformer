# Large Language Model v0.18 *Experimental*
import numpy as np
import random
import pickle

# Constants
generate_len = 100
dictionary_memory_uncompressed = 580
hidden_size = 1740
epochs = 50
compendium_filename = f"Compendium#{random.randint(0, 10000000)}.txt"
file = "test.txt"
word_dict_file = "word_dict.dat"

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        seq_len = inputs.shape[1]
        h_next = h_prev

        # Calculate memory accuracy for each time step
        outputs = np.zeros((self.output_size, seq_len))
        for t in range(seq_len):
            x_t = inputs[:, t].reshape(-1, 1)
            h_next = np.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_next) + self.b_h)
            y_t = np.dot(self.W_hy, h_prev) + self.b_y
            outputs[:, t] =  y_t.squeeze()
            h_prev = h_next
        return outputs, h_next

def softmax(x):
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
            pass
    return results


def generate_text_rnn(rnn, user_input, word_dict, length_to_generate):
    # Generate text using the RNN model
    generated_text = user_input[:]
    h_prev = np.zeros((hidden_size, 1))
    input_vector = np.zeros((len(word_dict), 1))

    # Initialize the input vector with the user input
    for word in user_input:
        input_indexes = find_word_index(word_dict, word)
        for input_index in input_indexes:
            input_vector[input_index] = 1


    for t in range(length_to_generate):
        # Forward pass with memory accuracy
        outputs, h_next = rnn.forward(input_vector, h_prev)
        adjusted_probabilities = softmax(outputs.flatten())

        rng = np.random.default_rng()
        next_index = rng.choice(len(adjusted_probabilities), p=adjusted_probabilities)

        # Get the next word from the word_dict
        next_word = word_dict[next_index]
        generated_text.append(next_word)

        # Prepare the input vector for the next step
        input_vector[next_index] = 1
        h_prev = h_next
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
        try:
            word_dict = load_word_dict(word_dict_file)
        except FileNotFoundError:
            word_dict = None

        input_size = len(word_dict)
        output_size = len(word_dict)
        rnn = RNN(input_size, hidden_size, output_size)

    if word_dict is None or _choice_ == "s":
        with open(file, encoding="UTF-8") as f:
            text = f.read()
        print("Learning from:", file)
        word_dict = preprocess_text(text)[:dictionary_memory_uncompressed]
        save_word_dict(word_dict, word_dict_file)

        input_size = len(word_dict)
        output_size = len(word_dict)
        rnn = RNN(input_size, hidden_size, output_size)
        h_prev = np.zeros((hidden_size, 1))

    while True:
        u_input = input("USER: ").strip().lower().split()
        rnn_generated_text = generate_text_rnn(rnn, u_input, word_dict, generate_len).lower()
        print("\nAI:", rnn_generated_text, "\n\n")
        with open(compendium_filename, "a", encoding="utf8") as f:
            f.write(f"\nAnswering: {' '.join(u_input)}\n{rnn_generated_text}\n")

if __name__ == "__main__":
    main()
