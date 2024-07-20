# Large Language Model v8.7 - George W

import numpy as np
import pickle
import re

KB_memory_uncompressed = -1  # KB access, -1 for unlimited
generate_length = 100
padding_token = '<unk>'

class NgramProcessor:
    def __init__(self, word_to_idx, padding_token):
        self.word_to_idx = word_to_idx
        self.padding_token = padding_token

    def get_partial_ngram_indices(self, ngram):
        words = ngram.split()  # Split ngram into individual words
        partial_ngrams = []

        # Generate all possible contiguous sub-sequences (partial ngrams)
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                partial_ngram = words[i:j]
                partial_ngrams.append(partial_ngram)

        # Convert words in each partial ngram to their indices
        all_indices = []
        for partial_ngram in partial_ngrams:
            indices = []
            for word in partial_ngram:
                idx = self.word_to_idx.get(word, self.word_to_idx.get(self.padding_token))
                if idx is not None:
                    indices.append(idx)
            if indices:
                all_indices.append(indices)

        return all_indices

class LanguageModel:
    def __init__(self, n=3, spill_factor=0.1):
        self.n = n
        self.spill_factor = spill_factor
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.matrix = None

    def create_ngrams(self, text):
        words = text.split()
        ngrams = zip(*[words[i:] for i in range(self.n)])
        return [' '.join(ngram) for ngram in ngrams]

    def is_valid_ngram(self, ngram):
        return re.match("^[a-zA-Z\s,]*$", ngram) is not None

    def encode_sentence(self, sentence):
        ngrams = self.create_ngrams(sentence)
        encoded = np.zeros(len(self.word_to_idx))
        processor = NgramProcessor(self.word_to_idx, padding_token)

        for ngram in ngrams:
            partial_ngram_indices = processor.get_partial_ngram_indices(ngram)
            for indices in partial_ngram_indices:
                for idx in indices:
                    if idx < len(encoded):
                        encoded[idx] += 1

        encoded_sum = encoded.sum()
        if encoded_sum > 0:
            encoded /= encoded_sum  # Normalize the encoded array to ensure it sums to 1
        return encoded

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))  # Numerical stability
        return exps / np.sum(exps)

    def compute_matrix(self, training_data):
        num_indices = len(self.word_to_idx)
        self.matrix = np.zeros((num_indices))

        # Compute the spill matrix based on training data
        for i, sentence in enumerate(training_data):
            ngrams = self.create_ngrams(sentence)
            processor = NgramProcessor(self.word_to_idx, padding_token)
            for ngram in ngrams:
                partial_ngram_indices = processor.get_partial_ngram_indices(ngram)
                for indices in partial_ngram_indices:
                    for idx in indices:
                        if 0 <= idx < num_indices:
                            if idx > 0:
                                self.matrix[:idx] -= self.spill_factor
                            if idx < num_indices - 1:
                                self.matrix[idx:] -= self.spill_factor
            if i % 1000 == 0:
                print("training:", i, "/", len(training_data))
        print("training:", len(training_data), "/", len(training_data))

    def train(self, filename):
        with open(filename, encoding="UTF-8") as f:
            training_data = f.read().lower().split(".")[:KB_memory_uncompressed]

        # Compute spill matrix based on training data
        self.compute_matrix(training_data)

    def chat(self, question):
        output = []
        input_seq = self.encode_sentence(question)
        probabilities = self.softmax(input_seq).flatten()

        if self.matrix is None:
            raise ValueError("Spill matrix is not computed. Please train the model first.")

        for t in range(generate_length):
            spilled_probabilities = np.dot(self.matrix, probabilities) + probabilities
            spilled_probabilities = np.maximum(spilled_probabilities, 1e-8)  # Avoid division by zero
            spilled_probabilities /= spilled_probabilities.sum()

            predicted_idx = np.random.choice(len(spilled_probabilities), p=spilled_probabilities)
            word = self.idx_to_word.get(predicted_idx, padding_token)
            output.append(word)

            input_seq = self.encode_sentence(word)  # Update input sequence based on generated words
            probabilities = self.softmax(input_seq)

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

    def save_model(self, filename):
        model_state = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'matrix': self.matrix
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_state = pickle.load(f)
        self.word_to_idx = model_state['word_to_idx']
        self.idx_to_word = model_state['idx_to_word']
        self.matrix = model_state['matrix']
        print(f"Model loaded from {filename}")

    def preprocess_data(self, filename):
        with open(filename, encoding="UTF-8") as f:
            conversations = f.read().lower().split(".")[:KB_memory_uncompressed]

        vocab = set()

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

if __name__ == "__main__":
    model = LanguageModel()

    _choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

    if _choice_ == "s":
        model.preprocess_data("test.txt")
        model.train("test.txt")
        model.save_model("model.dat")

    elif _choice_ == "l":
        model.load_model("model.dat")

    while True:
        user_input = input("You: ")
        response = model.chat(user_input)
        print(f"AI: {response}")
