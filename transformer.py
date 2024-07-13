
# Large Language Model v2.1 *Experimental*
import numpy as np
import math
import pickle

# Model parameters
hidden_size = 360 #last model saved requirement
dictionary_memory_uncompressed = 180 # KB access
learning_rate = 0.1
epochs = 5
generate_length = 100
padding_token = '<unk>'
model_file = "model.dat"
n = 3

# Create n-grams
def create_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

# Encoding function with <unk> token handling and PMI inclusion
def encode_sentence(sentence, word_to_idx, n):
    encoded = np.zeros(vocab_size)
    ngrams = create_ngrams(sentence, n)
    for ngram in ngrams:
        if ngram in word_to_idx:
            encoded[word_to_idx[ngram] - 1] = 1  # Use PMI value or default to 1.0 if not found
        else:
            encoded[word_to_idx[padding_token] - 1] = 0  #   # Assign 1.0 for <unk> token if n-gram is unknown
    return encoded

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.exp(exps)

class SimpleChatbotNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

        # Attention parameters
        self.Wa = np.random.randn(hidden_size, hidden_size)
        self.ba = np.zeros(hidden_size)
        self.v = np.random.randn(hidden_size)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def attention(self, hidden_states):
        # Compute attention scores
        attention_scores = np.inner(np.tanh(np.dot(hidden_states, self.Wa) + self.ba), self.v)
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=0, keepdims=True)
        context_vector = np.sum(attention_weights[:, np.newaxis] * hidden_states, axis=0)
        return context_vector

    def forward(self, x):
        self.hidden = np.dot(x, self.W1) + self.ba
        self.hidden_activation = np.tanh(self.hidden)

        # Apply attention
        context_vector = self.attention(self.hidden_activation)
        context_vector = precision_shift( context_vector, int(np.sum(x)))

        self.output = np.dot(context_vector, self.W2) + self.b2
        self.output_probs = np.exp(self.output) / np.sum(np.exp(self.output), axis=-1, keepdims=True)
        return self.output_probs

    def backward(self, x, target, output):
        d_output = output.copy()

        dW2 = np.outer(self.attention(self.W2.T), d_output)
        db2 = d_output

        d_hidden_activation = np.dot(d_output, self.W1)
        d_hidden = d_hidden_activation * (1 - np.power(self.hidden_activation, 2))

        dW1 = np.outer(x, d_hidden)
        db1 = d_hidden
        for dparam in [self.W1, self.b1, self.W2, self.b2]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update weights
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1.sum(axis=0)
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2.sum(axis=0)

    def train(self, x, target):
        output = self.forward(x)
        self.backward(x, target, output)

    def predict(self, x):
        output = self.forward(x)
        return output

    def save_model(self, filename):
        model_params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'Wa': self.Wa,
            'ba': self.ba,
            'v': self.v,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        self.W1 = model_params['W1']
        self.b1 = model_params['b1']
        self.W2 = model_params['W2']
        self.b2 = model_params['b2']
        self.Wa = model_params['Wa']
        self.ba = model_params['ba']
        self.input_size = model_params['input_size']
        self.hidden_size = model_params['hidden_size']
        self.output_size = model_params['output_size']
        print(f"Model loaded from {filename}")

def roll_encoded_sentence(encoded_sentence):
    return np.roll(encoded_sentence, -1)

def precision_shift(encoded_sentence, shift_size):
    return np.roll(encoded_sentence, shift_size)

def chat(model, question, generate_length, n):
    input_seq = encode_sentence(question, word_to_idx, n).reshape(1, -1)
    output = []

    for i in range(generate_length):
        idxs = model.predict(input_seq)
        adjusted_probabilities = softmax(idxs.flatten())

        # Invert the adjusted probabilities
        inverted_probabilities = 1 / adjusted_probabilities
        inverted_probabilities /= inverted_probabilities.sum()  # Normalize to ensure they sum to 1

        rng = np.random.default_rng()
        predicted_idx = rng.choice(range(len(inverted_probabilities)), p=inverted_probabilities)
        if predicted_idx + 1 in idx_to_word:  # Adjust index to start from 0
            output.append(idx_to_word[predicted_idx + 1])
        else:
            output.append(padding_token)

        last_ngram = output[-1].split()[-(n-1):]
        new_ngram = ' '.join(last_ngram + [idx_to_word[predicted_idx + 1]])  # Adjust index to start from 0
        input_seq = encode_sentence(new_ngram, word_to_idx, n)[:, np.newaxis].T

    return ' '.join(output)

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
        conversations = f.read().lower().split(".")[:dictionary_memory_uncompressed]


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

    vocab_size = len(vocab)
    output_size = vocab_size
    input_size = vocab_size

    model = SimpleChatbotNN(input_size, hidden_size, output_size)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for conv in conversations:
            input_seq = encode_sentence(conv, word_to_idx, n)
            target_seq = roll_encoded_sentence(encode_sentence(conv, word_to_idx, n))

            model.train(input_seq.reshape(1, -1), target_seq.reshape(1, -1))
            total_loss += np.sum((model.forward(input_seq.reshape(1, -1)) - target_seq)**2)

        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss}")

    model.save_model(model_file)

if (_choice_ == "l"):
    word_to_idx = load_word_dict("langA.dat")
    idx_to_word = load_word_dict("langB.dat")
    input_size = len(word_to_idx)
    output_size = len(word_to_idx)
    vocab_size = output_size
    model = SimpleChatbotNN(input_size, hidden_size, output_size)
    model.load_model(model_file)

# Example usage
while True:
    user_input = input("You: ")
    response = chat(model, user_input, generate_length, n)
    print(f"AI: {response}")
