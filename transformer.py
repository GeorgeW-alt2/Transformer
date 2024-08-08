# LLM v33.0
import numpy as np
import pickle
import re
import cmath

# Model parameters
KB_MEMORY_UNCOMPRESSED = -1  # -1 for unlimited
GENERATE_LENGTH = 50
PADDING_TOKEN = '<unk>'
N = 3

def filter_text(text):
    return re.sub(r'[^A-Za-z\s]', '', text)

def create_ngrams_and_words(text):
    words = text.split()
    return [' '.join(ngram) for n in range(1, N + 1)
            for ngram in zip(*[words[i:] for i in range(n)])]

def softmax(logits, target=None):
    # Calculate the magnitudes of the complex logits
    magnitudes = np.array([abs(cmath.phase(logit)) for logit in logits])

    # Exponentiate the magnitudes (as softmax typically applies the exponent)
    exps = np.exp(logits - np.max(magnitudes))  # Stability improvement by subtracting the max

    # Return the normalized probabilities
    return exps / np.sum(exps)

def text_to_vector(text, word_to_idx):
    vector = np.zeros(len(word_to_idx))
    tokens = create_ngrams_and_words(text)
    for token in tokens:
        vector[word_to_idx.get(token, word_to_idx[PADDING_TOKEN])] = 1
    return vector

def chat(question, word_to_idx, idx_to_word, generate_length, n):
    input_vector = text_to_vector(question, word_to_idx)
    input_vectorB = text_to_vector(question, word_to_idx)

    output = []
    for _ in range(generate_length):
        probabilities = softmax(input_vectorB.flatten(), input_vector.flatten())
        predicted_idx = np.random.choice(range(len(probabilities)), p=probabilities)

        ngram = idx_to_word.get(predicted_idx, PADDING_TOKEN)
        output.append(ngram)
        input_vector = text_to_vector(' '.join(output), word_to_idx)

    # Backward chaining: refine the output in reverse
    for i in range(len(output) - 1, 0, -1):
        refined_input = ' '.join(output[:i])
        input_vector = text_to_vector(refined_input, word_to_idx)
        probabilities = softmax(input_vector.flatten())
        chosen_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        output[i-2] = idx_to_word.get(chosen_idx, PADDING_TOKEN)

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

_choice_ = input("\nSave new model/Load old model? [s/l]: ").lower()
word_to_idx = idx_to_word = {}

if _choice_ == "s":
    with open("test.txt", encoding="UTF-8") as f:
        conversations = remove_sentences_with_numbers_and_symbols(f.read().lower().split(".")[:KB_MEMORY_UNCOMPRESSED])
    vocab = list(set(ngram for conv in conversations for ngram in create_ngrams_and_words(conv + ".")))
    vocab.append(PADDING_TOKEN)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    save_dict(word_to_idx, "langA.dat")
    save_dict(idx_to_word, "langB.dat")
    centers = np.linspace(-1, 1, len(word_to_idx))
    
elif _choice_ == "l":
    word_to_idx = load_dict("langA.dat")
    idx_to_word = load_dict("langB.dat")
    centers = np.linspace(-1, 1, len(word_to_idx))

while True:
    user_input = filter_text(input("You: "))
    response_begin = chat(user_input, word_to_idx, idx_to_word, GENERATE_LENGTH, N)
    print(f"AI: {response_begin}\n")
