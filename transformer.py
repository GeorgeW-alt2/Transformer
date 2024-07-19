# Large Language Model v7.2 - George W

import numpy as np
import pickle
import re

KB_memory_uncompressed = -1  # KB access, -1 for unlimited
generate_length = 100
padding_token = '<unk>'

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total

    def range_query(self, left, right):
        return self.query(right) - self.query(left - 1)

class LanguageModel:
    def __init__(self, n=3, D=200, learning_rate=0.01):
        self.n = n
        self.D = D
        self.learning_rate = learning_rate
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.W = None
        self.b = None
        self.cells = np.random.randn(D, D)  # Added cells for RFF
        self.alpha = 0.5  # Multinomial emission parameter
        self.fenwick_tree = None  # Fenwick Tree for managing n-gram counts

    def create_ngrams(self, text):
        words = text.split()
        ngrams = zip(*[words[i:] for i in range(self.n)])
        return [' '.join(ngram) for ngram in ngrams]

    def is_valid_ngram(self, ngram):
        return re.match("^[a-zA-Z\s,]*$", ngram) is not None

    def encode_sentence(self, sentence):
        encoded = np.zeros(len(self.word_to_idx))
        ngrams = self.create_ngrams(sentence)
        for ngram in ngrams:
            idx = self.word_to_idx.get(ngram, self.word_to_idx.get(padding_token))
            if idx is not None:
                encoded[idx - 1] = 1
        return encoded

    def hsa_mapping(self, input_vec):
        z = np.dot(self.W, input_vec) + self.b
        hsa_transformed = np.zeros_like(z)

        # Hierarchical Softmax Activation with Fenwick Tree adjustment
        for i in range(len(z)):
            ngram_idx = i + 1
            ngram_count = self.query_ngram_count(ngram_idx)  # Get n-gram count from Fenwick Tree
            hsa_transformed[i] = np.log(1 + np.exp(z[i] - self.alpha * ngram_count))  # Adjust activation based on count

        return hsa_transformed.flatten()

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))  # Numerical stability
        return exps / np.sum(exps)

    def chat(self, question):
        output = []
        input_seq = self.encode_sentence(question)
        hsa_input = self.hsa_mapping(input_seq)

        for _ in range(generate_length):
            probabilities = self.softmax(hsa_input.flatten())

            # Adjust probabilities using Fenwick Tree
            adjusted_probabilities = np.copy(probabilities)
            for i in range(len(probabilities)):
                ngram_idx = i + 1
                ngram_count = self.query_ngram_count(ngram_idx)
                adjusted_probabilities[i] = probabilities[i] * (1 + ngram_count / 10.0)  # Example adjustment

            adjusted_probabilities /= adjusted_probabilities.sum()  # Normalize adjusted probabilities

            predicted_idx = np.random.choice(len(adjusted_probabilities), p=adjusted_probabilities)
            word = self.idx_to_word.get(predicted_idx + 1, padding_token)
            output.append(word)

            input_seq = self.encode_sentence(word)
            hsa_input = self.hsa_mapping(input_seq)

        return ' '.join(output)

    def save_word_dict(self, word_dict, filename):
        with open(filename, 'wb') as f:
            pickle.dump(word_dict, f)
        print(f"Dictionary saved to {filename}")

    def load_word_dict(self, filename):
        with open(filename, 'rb') as f:
            word_dict = pickle.load(f)
        print(f"Dictionary loaded from {filename}")
        return word_dict

    def preprocess_data(self, filename):
        with open(filename, encoding="UTF-8") as f:
            conversations = f.read().lower().split(".")[:KB_memory_uncompressed]

        vocab = set()
        ngram_indices = []

        for conv in conversations:
            ngrams = self.create_ngrams(conv)
            for ngram in ngrams:
                if self.is_valid_ngram(ngram):
                    vocab.add(ngram)

        vocab.add(padding_token)

        self.word_to_idx = {word: idx for idx, word in enumerate(vocab, start=1)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.save_word_dict(self.word_to_idx, "langA.dat")
        self.save_word_dict(self.idx_to_word, "langB.dat")

        # Initialize and update Fenwick Tree with n-gram counts
        self.initialize_fenwick_tree(len(self.word_to_idx))
        for idx in range(1, len(self.word_to_idx) + 1):
            self.update_ngram_counts(idx, 1)  # Example update with count 1

    def initialize_fenwick_tree(self, size):
        self.fenwick_tree = FenwickTree(size)

    def update_ngram_counts(self, ngram_idx, delta):
        if self.fenwick_tree:
            self.fenwick_tree.update(delta,ngram_idx)

    def query_ngram_count(self, ngram_idx):
        if self.fenwick_tree:
            return self.fenwick_tree.query(ngram_idx)
        return 0

if __name__ == "__main__":
    model = LanguageModel()

    _choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

    if _choice_ == "s":
        model.preprocess_data("test.txt")

    if _choice_ == "l":
        model.word_to_idx = model.load_word_dict("langA.dat")
        model.idx_to_word = model.load_word_dict("langB.dat")

    linear_space_array = np.linspace(-1, 1, model.D * len(model.word_to_idx))  # Adjust the range as needed
    model.W = linear_space_array.reshape(model.D, len(model.word_to_idx))
    model.b = np.linspace(0, 2 * np.pi, model.D)
    while True:
        user_input = input("You: ")
        response = model.chat(user_input)
        print(f"AI: {response}")
