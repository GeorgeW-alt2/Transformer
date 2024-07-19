# Large Language Model v6.9 - George W

import numpy as np
import pickle
import re

KB_memory_uncompressed = -1  # KB access, -1 for unlimited
generate_length = 100
padding_token = '<unk>'

class LanguageModel:
    def __init__(self, n=3, D=200, learning_rate=0.01):
        self.n = n
        self.D = D
        self.learning_rate = learning_rate
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.W = None
        self.b = None

    def create_ngrams(self, text):
        words = text.split()
        ngrams = zip(*[words[i:] for i in range(self.n)])
        return [' '.join(ngram) for ngram in ngrams]

    def is_valid_ngram(self, ngram):
        return re.match("^[a-zA-Z0-9\s.,]*$", ngram) is not None

    def encode_sentence(self, sentence):
        encoded = np.zeros(len(self.word_to_idx))
        ngrams = self.create_ngrams(sentence)
        for ngram in ngrams:
            idx = self.word_to_idx.get(ngram, self.word_to_idx.get(padding_token))
            if idx is not None:
                encoded[idx - 1] = 1
        return encoded

    def tbi_mapping(self, input_vec):
        z = np.dot(self.W, input_vec) + self.b
        tbi_transformed = np.zeros_like(z)

        # Define triangular basis functions
        for i in range(len(z)):
            for j in range(self.D):
                center = (j + 0.5) / self.D
                width = 1.0 / self.D
                tbi_transformed[i] += np.maximum(0, 1 - np.abs((z[i] - center) / width))

        return tbi_transformed.flatten()

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))  # Numerical stability
        return exps / np.sum(exps)

    def chat(self, question):
        output = []
        input_seq = self.encode_sentence(question)
        tbi_input = self.tbi_mapping(input_seq)

        for _ in range(generate_length):
            probabilities = self.softmax(tbi_input.flatten())

            # Dynamic bias adjustment based on sentence length or context
            sentence_length = len(output) + 1
            context_bias = np.exp(-0.1 * np.arange(len(probabilities)))  # Example bias decay
            biased_probabilities = probabilities * context_bias
            biased_probabilities /= biased_probabilities.sum()  # Normalize

            predicted_idx = np.random.choice(len(biased_probabilities), p=biased_probabilities)
            word = self.idx_to_word.get(predicted_idx + 1, padding_token)
            output.append(word)

            input_seq = self.encode_sentence(next_input)
            tbi_input = self.tbi_mapping(input_seq)

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

        linear_space_array = np.linspace(-1, 1, self.D * len(self.word_to_idx))  # Adjust the range as needed
        self.W = linear_space_array.reshape(self.D, len(self.word_to_idx))
        self.b = np.linspace(0, 2 * np.pi, self.D)

if __name__ == "__main__":
    model = LanguageModel()

    _choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

    if _choice_ == "s":
        model.preprocess_data("test.txt")

    if _choice_ == "l":
        model.word_to_idx = model.load_word_dict("langA.dat")
        model.idx_to_word = model.load_word_dict("langB.dat")

    while True:
        user_input = input("You: ")
        response = model.chat(user_input)
        print(f"AI: {response}")
