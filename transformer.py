#Large Language Model v6.0 - George W
import numpy as np
import pickle
import re

KB_memory_uncompressed = -1  # KB access, -1 for unlimited
generate_length = 100
padding_token = '<unk>'

class LanguageModel:
    def __init__(self, n=3, D=200):
        self.n = n
        self.D = D
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.W = None
        self.b = None
        self.sperner_families = []

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
            if ngram in self.word_to_idx:
                encoded[self.word_to_idx[ngram] - 1] = 1
            else:
                encoded[self.word_to_idx[padding_token] - 1] = 0
        return encoded

    def generate_sperner_families(self, k):
        # Generates k Sperner families for use in RFF
        self.sperner_families = []
        for i in range(k):
            family = set()
            elements = np.random.choice(self.D, size=np.random.randint(1, self.D), replace=False)
            for element in elements:
                family.add(frozenset([element]))
            self.sperner_families.append(family)

    def rgf_mapping(self, input_vec):
        z = np.dot(self.W, input_vec) + self.b
        base_transformed = np.sqrt(2 / self.D) * np.concatenate((np.cos(z), np.sin(z)))

        # Modify the base_transformed vector using Sperner families
        for family in self.sperner_families:
            for subset in family:
                indices = list(subset)
                if len(indices) > 1:
                    base_transformed[indices] = base_transformed[indices].sum()

        return base_transformed

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        return exps / np.sum(exps)

    def chat(self, question):
        output = []
        input_seq = self.encode_sentence(question)
        rgf_input = self.rgf_mapping(input_seq)

        for i in range(generate_length):
            adjusted_probabilities = self.softmax(rgf_input.flatten())

            # Invert the adjusted probabilities
            inverted_probabilities = 1 / adjusted_probabilities
            inverted_probabilities /= inverted_probabilities.sum()  # Normalize to ensure they sum to 1

            rng = np.random.default_rng()
            predicted_idx = rng.choice(range(len(inverted_probabilities)), p=inverted_probabilities)
            if predicted_idx + 1 in self.idx_to_word:  # Adjust index to start from 0
                output.append(self.idx_to_word[predicted_idx + 1])
            else:
                output.append(padding_token)

            next_input = ' '.join(output)
            input_seq = self.encode_sentence(next_input)
            rgf_input = self.rgf_mapping(input_seq)

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

    def save_rgf_params(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.W, self.b), f)
        print(f"RGF parameters saved to {filename}")

    def load_rgf_params(self, filename):
        with open(filename, 'rb') as f:
            self.W, self.b = pickle.load(f)
        print(f"RGF parameters loaded from {filename}")
        return self.W, self.b

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

        # Generate a linearly spaced array for self.b
        self.b = np.linspace(0, 2 * np.pi, self.D)
        self.generate_sperner_families(100)  # Generate 10 Sperner families for example
        self.save_rgf_params("rgf_params.dat")

if __name__ == "__main__":
    model = LanguageModel()

    _choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

    if _choice_ == "s":
        model.preprocess_data("test.txt")

    if _choice_ == "l":
        model.word_to_idx = model.load_word_dict("langA.dat")
        model.idx_to_word = model.load_word_dict("langB.dat")
        model.W, model.b = model.load_rgf_params("rgf_params.dat")

    while True:
        user_input = input("You: ")
        response = model.chat(user_input)
        print(f"AI: {response}")
