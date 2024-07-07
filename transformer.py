# Large Language Model v0.2 *Experimental*
import numpy as np
import random
import pickle

# Constants
generate_len = 100
dictionary_memory_uncompressed = 580
hidden_size = 740
epochs = 5
compendium_filename = f"Compendium#{random.randint(0, 10000000)}.txt"
file = "test.txt"
word_dict_file = "word_dict.dat"
model_file = "model.dat"

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

    def save_model(self, filename):
        model_params = {
            'W_xh': self.W_xh,
            'W_hh': self.W_hh,
            'W_hy': self.W_hy,
            'b_h': self.b_h,
            'b_y': self.b_y,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        self.W_xh = model_params['W_xh']
        self.W_hh = model_params['W_hh']
        self.W_hy = model_params['W_hy']
        self.b_h = model_params['b_h']
        self.b_y = model_params['b_y']
        self.input_size = model_params['input_size']
        self.hidden_size = model_params['hidden_size']
        self.output_size = model_params['output_size']
        self.learning_rate = model_params['learning_rate']
        print(f"Model loaded from {filename}")

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

    def backward(self, inputs, targets, h_prev, outputs):
        dW_xh, dW_hh, dW_hy = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy)
        db_h, db_y = np.zeros_like(self.b_h), np.zeros_like(self.b_y)
        dh_next = np.zeros_like(h_prev)

        for t in reversed(range(len(inputs[0]))):
            dy = outputs[:, t].reshape(-1, 1) - targets[:, t].reshape(-1, 1)
            dW_hy += np.dot(dy, h_prev.T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - h_prev ** 2) * dh
            db_h += dh_raw
            dW_xh += np.dot(dh_raw, inputs[:, t].reshape(1, -1))
            dW_hh += np.dot(dh_raw, h_prev.T)
            dh_next = np.dot(self.W_hh.T, dh_raw)

        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)

        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_h -= self.learning_rate * db_h
        self.b_y -= self.learning_rate * db_y


    def train(self, inputs, targets, epochs=50):
        for epoch in range(epochs):
            h_prev = np.zeros((self.hidden_size, 1))
            outputs, h_states = self.forward(inputs, h_prev)
            self.backward(inputs, targets, h_states, outputs)
            print(f'Epoch {epoch+1}/{epochs}')

    def calculate_loss(self, outputs, targets):
        loss = np.sum((outputs - targets) ** 2) / 2
        return loss

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
    rnn = None
    if (_choice_ == "l"):
        try:
            word_dict = load_word_dict(word_dict_file)
            input_size = len(word_dict)
            output_size = len(word_dict)
            rnn = RNN(input_size, hidden_size, output_size)
            rnn.load_model(model_file)
        except FileNotFoundError:
            word_dict = None

    if (word_dict is None or _choice_ == "s"):
        with open(file, encoding="UTF-8") as f:
            text = f.read()
        print("Learning from:", file)
        word_dict = preprocess_text(text)[:dictionary_memory_uncompressed]
        save_word_dict(word_dict, word_dict_file)

        input_size = len(word_dict)
        output_size = len(word_dict)
        rnn = RNN(input_size, hidden_size, output_size)
        h_prev = np.zeros((hidden_size, 1))

        inputs = np.zeros((input_size, len(word_dict)))
        targets = np.zeros((output_size, len(word_dict)))
        for i, word in enumerate(word_dict):
            input_indexes = find_word_index(word_dict, word)
            for input_index in input_indexes:
                inputs[input_index, i] = 1
            targets[:, i] = np.roll(inputs[:, i], -1)

        rnn.train(inputs, targets, epochs)
        rnn.save_model(model_file)

    while True:
        u_input = input("USER: ").strip().lower().split()
        rnn_generated_text = generate_text_rnn(rnn, u_input, word_dict, generate_len).lower()
        print("\nAI:", rnn_generated_text, "\n\n")
        with open(compendium_filename, "a", encoding="utf8") as f:
            f.write(f"\nAnswering: {' '.join(u_input)}\n{rnn_generated_text}\n")

if __name__ == "__main__":
    main()
