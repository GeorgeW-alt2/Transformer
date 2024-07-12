#Poloniex bot v0.1 - George W - 2024
import hashlib
import urllib
import urllib.parse
import urllib.request
import requests
import time
import hmac
import base64
import json

# Model parameters
hidden_size = 660 #last model saved requirement
learning_rate = 0.1
epochs = 15
generate_length = 10
padding_token = '<unk>'
model_file = "model.dat"
n = 3

access_key = ""
secret_key = ""

symbol = "BTC_USDT"

limit_TH = 1000
class SDK:

    def __init__(self, access_key, secret_key):
        self.__access_key = access_key
        self.__secret_key = secret_key

    def __create_sign(self, params, method, path):
        timestamp = int(time.time() * 1000)
        if method.upper() == "GET":
            params.update({"signTimestamp": timestamp})
            sorted_params = sorted(params.items(), key=lambda d: d[0], reverse=False)
            encode_params = urllib.parse.urlencode(sorted_params)
            del params["signTimestamp"]
        else:
            requestBody = json.dumps(params)
            encode_params = "requestBody={}&signTimestamp={}".format(requestBody, timestamp)
        sign_params_first = [method.upper(), path, encode_params]
        sign_params_second = "\n".join(sign_params_first)
        sign_params = sign_params_second.encode(encoding="UTF8")
        secret_key = self.__secret_key.encode(encoding="UTF8")
        digest = hmac.new(secret_key, sign_params, digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(digest)
        signature = signature.decode()
        return signature, timestamp

    def sign_req(self, host, path, method, params, headers):
        sign, timestamp = self.__create_sign(params=params, method=method, path=path)
        headers.update({
            "key": self.__access_key,
            "signTimestamp": str(timestamp),
            "signature": sign,
        })

        url = f"{host}{path}"

        try:
            if method.upper() == "POST":
                response = requests.post(url, data=json.dumps(params), headers=headers)
            elif method.upper() == "GET":
                params_encoded = urllib.parse.urlencode(params)
                response = requests.get(f"{url}?{params_encoded}", headers=headers)
            elif method.upper() == "PUT":
                response = requests.put(url, data=json.dumps(params), headers=headers)
            elif method.upper() == "DELETE":
                response = requests.delete(url, data=json.dumps(params), headers=headers)
            else:
                raise ValueError("Invalid HTTP method")

            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                print("Error decoding JSON. Response content:")
                print(response.text)
                raise

        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            if response is not None:
                print("Response content:")
                print(response.text)
            raise

    def place_order(self, symbol, side, price, quantity):
        path = "/orders"
        method = "POST"
        params = {
            "symbol": symbol,
            "type": "limit",
            "side": side,
            "timeInForce": "GTC",
            "price": str(price),
            "amount": str(quantity),
            "quantity": str(quantity),
            "clientOrderId": "",
        }
        headers = {"Content-Type": "application/json"}
        response = self.sign_req(host, path, method, params, headers)
        print(f"Order {side} placed at price {price}")

        return response

    def get_trade_history(self, host, symbol, limit):
        path = f"/markets/{symbol}/trades"
        method = "GET"
        params = {
            "limit": limit
        }
        headers = {"Content-Type": "application/json"}
        response = self.sign_req(host, path, method, params, headers)

        # Extract prices into an array
        prices = [item['price'] for item in response]

        return prices

def detect_peaks_valleys(prices):
    peaks = []
    valleys = []

    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            peaks.append(i)
        elif prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
            valleys.append(i)

    return peaks, valleys

# Example auto trade logic
def auto_trade(symbol, limit):
    prices = service.get_trade_history(host, symbol, limit)
    valleys, peaks = detect_peaks_valleys(prices)

    if valleys:
        # Example buy logic - place a buy order at the lowest valley price
        lowest_valley_index = valleys[0]
        lowest_valley_price = prices[lowest_valley_index]
        print(f"Buying at {lowest_valley_price}")

        # Implement your order placement logic here
        # Example:
        # place_order(symbol, "buy", lowest_valley_price, quantity)

    if peaks:
        # Example sell logic - place a sell order at the highest peak price
        highest_peak_index = peaks[-1]
        highest_peak_price = prices[highest_peak_index]
        print(f"Selling at {highest_peak_price}")

        # Implement your order placement logic here
        # Example:
        # place_order(symbol, "sell", highest_peak_price, quantity)

if __name__ == "__main__":

    headers = {"Content-Type": "application/json"}
    host = "https://api.poloniex.com"
    service = SDK(access_key, secret_key)

    # Example: get trade history
    prices = service.get_trade_history(host, symbol, limit_TH)

#Neural network segment

# Large Language Model v1.9 *Experimental*
import numpy as np
import math
import pickle

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
        predicted_idx = rng.choice(range(len(inverted_probabilities)), p=roll_encoded_sentence(inverted_probabilities))
        input_seq = precision_shift(input_seq, predicted_idx)
        if predicted_idx + 1 in idx_to_word:  # Adjust index to start from 0
            output.append(idx_to_word[predicted_idx + 1])
        else:
            output.append(padding_token)

        last_ngram = output[-1].split()[-(n-1):]
        new_ngram = ' '.join(last_ngram + [idx_to_word[predicted_idx + 1]])  # Adjust index to start from 0

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
    data = ' '.join(prices)
    # Vocabulary creation including PMI values
    vocab = set()
    ngrams = create_ngrams(data, n)
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
        input_seq = encode_sentence(data, word_to_idx, n)
        target_seq = roll_encoded_sentence(encode_sentence(data, word_to_idx, n))

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
    input("Press enter to continue...")
    user_input = prices[-1]
    response = chat(model, user_input, generate_length, n)
    print(f"AI: {response}")
    symbol = "BTC_USDT"
    limitB = 100
    auto_trade(symbol, limitB)
    print()

