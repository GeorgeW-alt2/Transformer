# LLM v30.2
import numpy as np
import pickle
import re

# Model parameters
KB_MEMORY_UNCOMPRESSED = 100  # -1 for unlimited
GENERATE_LENGTH = 50
SIGMA = 0.7
PADDING_TOKEN = '<unk>'
ALPHA = 0


def filter_text(text):
    return re.sub(r'[^A-Za-z\s]', '', text)


def create_ngrams_and_words(text, max_n):
    words = text.split()
    return [' '.join(ngram) for n in range(1, max_n + 1)
            for ngram in zip(*[words[i:] for i in range(n)])]


def gaussian_rbf(x, c, s, alpha):
    alpha += 1
    return np.exp(-np.dot(-x.reshape(1, -1), x.reshape(-1, 1)) ** 2 / (2 * s ** 2))[-alpha]


def encode_ngram(ngram, token_vector, word_to_idx, centers, sigma, alpha):
    idx = word_to_idx.get(ngram, word_to_idx[PADDING_TOKEN])
    return idx, gaussian_rbf(token_vector, centers[idx], sigma, alpha)


def encode_sentence(sentence, word_to_idx, centers, sigma, max_n):
    tokens = create_ngrams_and_words(sentence, max_n)
    token_vector = np.zeros(len(word_to_idx))
    for token in tokens:
        token_vector[word_to_idx.get(token, word_to_idx[PADDING_TOKEN])] = 1
    encoded = np.zeros(len(word_to_idx))
    for token in tokens:
        idx, rbf_value = encode_ngram(token, token_vector, word_to_idx, centers, sigma, ALPHA)
        encoded[idx] = np.linalg.norm(idx) * np.linalg.norm(rbf_value)

    return encoded


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / magnitude if magnitude != 0 else 0


def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def text_to_vector(text, word_to_idx):
    vector = np.zeros(len(word_to_idx))
    tokens = create_ngrams_and_words(text, N)
    for token in tokens:
        vector[word_to_idx.get(token, word_to_idx[PADDING_TOKEN])] = 1
    return vector


def chat(ngram_encoding_index, question, word_to_idx, generate_length, n):
    input_vector = text_to_vector(question, word_to_idx)
    encoded = np.zeros(len(word_to_idx))
    for ngram in create_ngrams_and_words(question, n):
        if ngram in ngram_encoding_index:
            idx, rbf_value = ngram_encoding_index[ngram]
            encoded[idx] = rbf_value
    output = []
    for _ in range(generate_length):
        probabilities = softmax(encoded.flatten())
        predicted_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        ngram = idx_to_word.get(predicted_idx, PADDING_TOKEN)
        output.append(ngram)
        encoded = np.zeros(len(word_to_idx))
        next_input = ' '.join(output)
        for ngram in create_ngrams_and_words(next_input, n):
            if ngram in ngram_encoding_index:
                idx, rbf_value = ngram_encoding_index[ngram]
                encoded[idx] = rbf_value
    return ' '.join(output)


def save_dict(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
    print(f"Dictionary saved to {filename}")


def load_dict(filename):
    with open(filename, 'rb') as f:
        dictionary = pickle.load(f)
    print(f"Dictionary loaded from {filename}")
    return dictionary


def remove_sentences_with_numbers_and_symbols(sentences):
    return [s for s in sentences if re.match(r'^[A-Za-z\s,.]+$', s)]


def print_progress_bar(iteration, total, prefix='', length=50, fill='█'):
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% Complete', end='\r')
    if iteration == total:
        print()


_choice_ = input("\nSave new model/Load old model? [s/l]: ").lower()
word_to_idx = idx_to_word = ngram_encoding_index = {}

if _choice_ == "s":
    N = 3
    with open("test.txt", encoding="UTF-8") as f:
        conversations = remove_sentences_with_numbers_and_symbols(f.read().lower().split(".")[:KB_MEMORY_UNCOMPRESSED])
    vocab = list(set(ngram for conv in conversations for ngram in create_ngrams_and_words(conv + ".", N)))
    vocab.append(PADDING_TOKEN)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    save_dict(word_to_idx, "langA.dat")
    save_dict(idx_to_word, "langB.dat")
    centers = np.linspace(-1, 1, len(word_to_idx))
    total_ngrams = len(vocab)
    current_progress = 0
    print_progress_bar(current_progress, total_ngrams, prefix='AutoGen:')
    for ngram in vocab:
        token_vector = np.zeros(len(word_to_idx))
        token_vector[word_to_idx.get(ngram, word_to_idx[PADDING_TOKEN])] = 1
        idx, rbf_value = encode_ngram(ngram, token_vector, word_to_idx, centers, SIGMA, ALPHA)
        ngram_encoding_index[ngram] = (idx, rbf_value)
        current_progress += 1
        print_progress_bar(current_progress, total_ngrams, prefix='AutoGen:')
    save_dict(ngram_encoding_index, "model.dat")

elif _choice_ == "l":
    word_to_idx = load_dict("langA.dat")
    idx_to_word = load_dict("langB.dat")
    ngram_encoding_index = load_dict("model.dat")
    centers = np.linspace(-1, 1, len(word_to_idx))

N = 2
while True:
    user_input = filter_text(input("You: "))
    response_begin = chat(ngram_encoding_index, user_input, word_to_idx, GENERATE_LENGTH, N)
    print(f"AI: {response_begin}\n")
