# Large Language Model v7.4 - George W

import numpy as np
import pickle
import re

KB_memory_uncompressed = 1000  # KB access, -1 for unlimited
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
                indices.append(idx)
            all_indices.append(indices)

        return all_indices

class LanguageModel:
    def __init__(self, n=3, spill_factor=0.1):
        self.n = n
        self.spill_factor = spill_factor
        self.word_to_idx = {}
        self.idx_to_word = {}

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
                    encoded[idx] += 1

        encoded /= encoded.sum()  # Normalize the encoded array to ensure it sums to 1
        return encoded

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))  # Numerical stability
        return exps / np.sum(exps)

    def chat(self, question):
        output = []
        input_seq = self.encode_sentence(question)
        probabilities = self.softmax(input_seq).flatten()

        for t in range(generate_length):
            # Apply probability spill
            spilled_probabilities = probabilities.copy()
            for idx in range(len(spilled_probabilities)):
                if idx > 0:
                    spilled_probabilities[idx:] += self.spill_factor * probabilities[idx]
                if idx < len(spilled_probabilities) - 1:
                    spilled_probabilities[:idx] += self.spill_factor * probabilities[idx]

            # Normalize to ensure the probabilities sum to 1
            spilled_probabilities /= spilled_probabilities.sum()

            # Select the next word based on the adjusted probabilities
            predicted_idx = np.random.choice(len(spilled_probabilities), p=spilled_probabilities)
            word = self.idx_to_word.get(predicted_idx + 1, padding_token)
            output.append(word)

            # Update input sequence and prediction
            input_seq = self.encode_sentence(' '.join(output))
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

    if _choice_ == "l":
        model.word_to_idx = model.load_word_dict("langA.dat")
        model.idx_to_word = model.load_word_dict("langB.dat")

    while True:
        user_input = input("You: ")
        response = model.chat(user_input)
        print(f"AI: {response}")
