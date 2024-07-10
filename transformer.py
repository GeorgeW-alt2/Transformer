# Large Language Model v0.4 *Experimental*

import numpy as np
import random

# Model parameters
hidden_size = 160
dictionary_memory_uncompressed = 180
learning_rate = 0.1
epochs = 5
generate_length = 100
padding_token = '<unk>'

# Load and preprocess data
with open("test.txt", encoding="UTF-8") as f:
    conversations = f.read().split(".")[:dictionary_memory_uncompressed]

# Create n-grams
def create_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

# Vocabulary creation
n = 3  # Choose the n-gram size
vocab = set()
for conv in conversations:
    ngrams = create_ngrams(conv, n)
    vocab.update(ngrams)

# Add a special token for unknown words
vocab.add(padding_token)

word_to_idx = {word: idx for idx, word in enumerate(vocab, start=1)}  # Start indexing from 1
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

vocab_size = len(vocab)
output_size = vocab_size
input_size = vocab_size

# Encoding function with <unk> token handling
def encode_sentence(sentence, word_to_idx, n):
    encoded = np.zeros(vocab_size)
    ngrams = create_ngrams(sentence, n)
    for ngram in ngrams:
        if ngram in word_to_idx:
            encoded[word_to_idx[ngram]-1] = 1  # Adjust index to start from 0
        else:
            encoded[word_to_idx[padding_token]-1] = 1  # Assign <unk> token index if n-gram is unknown
    return encoded

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

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

    def attention(self, hidden_states):
        # Compute attention scores
        attention_scores = np.dot(np.tanh(np.dot(hidden_states, self.Wa) + self.ba), self.v)
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=0, keepdims=True)
        context_vector = np.sum(attention_weights[:, np.newaxis] * hidden_states, axis=0)
        return context_vector

    def forward(self, x):
        self.hidden = np.dot(x, self.W1) + self.b1
        self.hidden_activation = np.tanh(self.hidden)

        # Apply attention
        context_vector = self.attention(self.hidden_activation)

        self.output = np.dot(context_vector, self.W2) + self.b2
        self.output_probs = np.exp(self.output) / np.sum(np.exp(self.output), axis=-1, keepdims=True)
        return self.output_probs

    def backward(self, x, target, output):
        d_output = output.copy()

        dW2 = np.outer(self.attention(self.hidden_activation), d_output)
        db2 = d_output

        d_hidden_activation = np.dot(d_output, self.W2.T)
        d_hidden = d_hidden_activation * (1 - np.power(self.hidden_activation, 2))

        dW1 = np.dot(x.T, d_hidden)
        db1 = d_hidden

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

def roll_encoded_sentence(encoded_sentence):
    return np.roll(encoded_sentence, 1)

# Training loop
model = SimpleChatbotNN(input_size, hidden_size, output_size)

for epoch in range(epochs):
    total_loss = 0
    for conv in conversations:
        input_seq = encode_sentence(conv, word_to_idx, n)
        target_seq = roll_encoded_sentence(encode_sentence(conv, word_to_idx, n))

        model.train(input_seq.reshape(1, -1), target_seq.reshape(1, -1))
        total_loss += np.sum((model.forward(input_seq.reshape(1, -1)) - target_seq)**2)

    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss}")

def chat(model, question, generate_length, n):
    input_seq = encode_sentence(question, word_to_idx, n).reshape(1, -1)
    output = []

    for i in range(generate_length):
        idxs = model.predict(input_seq)
        adjusted_probabilities = softmax(idxs.flatten())

        rng = np.random.default_rng()
        predicted_idx = rng.choice(len(adjusted_probabilities), p=adjusted_probabilities)

        if predicted_idx in idx_to_word:
            output.append(idx_to_word[predicted_idx])
        else:
            output.append(padding_token)

        last_ngram = output[-1].split()[-(n-1):]
        new_ngram = ' '.join(last_ngram + [idx_to_word[predicted_idx]])
        input_seq = encode_sentence(new_ngram, word_to_idx, n).reshape(1, -1)

    return ' '.join(output)

# Example usage
while True:
    user_input = input("You: ")
    response = chat(model, user_input, generate_length, n)
    print(f"GPT: {response}")
