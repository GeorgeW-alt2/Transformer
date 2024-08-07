# LLM v31.0
import numpy as np
import pickle
import re

# Model parameters
KB_MEMORY_UNCOMPRESSED = -1  # -1 for unlimited
GENERATE_LENGTH = 50
SPILL_FACTOR=0.1
PADDING_TOKEN = '<unk>'
N = 3

def filter_text(text):
    return re.sub(r'[^A-Za-z\s]', '', text)

def create_ngrams_and_words(text, max_n):
    words = text.split()
    return [' '.join(ngram) for n in range(1, max_n + 1)
            for ngram in zip(*[words[i:] for i in range(n)])]

def encode_sentence(sentence, word_to_idx, centers, sigma, max_n):
    tokens = create_ngrams_and_words(sentence, max_n)
    token_vector = np.zeros(len(word_to_idx))
    for token in tokens:
        token_vector[word_to_idx.get(token, word_to_idx[PADDING_TOKEN])] = 1
    encoded = softmax(token_vector)

    return encoded
    
def softmax(logits):
    exps = np.exp((logits - np.max(logits)))
    return exps / np.sum(exps)
    
def softmax_with_gradients(logits):
    exps = np.exp(logits - np.max(logits))
    softmax_probs = exps / np.sum(exps)
    
    # Spillover the probabilities with a gradient
    spilled_probs = np.zeros_like(softmax_probs)
    for i in range(len(softmax_probs)):
        # Main probability with reduced spill factor
        spilled_probs[i] += softmax_probs[i] * (1 - spill_factor)
        
        # Gradient spill to adjacent elements
        for j in range(1, len(softmax_probs)):
            left_index = i+j
            right_index = i - j
            gradient_spill = SPILL_FACTOR / (j + 1)
            
            if left_index >= 0:
                spilled_probs[left_index] += softmax_probs[i] * gradient_spill
            if right_index < len(softmax_probs):
                spilled_probs[right_index] += softmax_probs[i] * gradient_spill
    
    # Normalize the probabilities to make sure they sum up to 1
    spilled_probs /= np.sum(spilled_probs)
    
    return spilled_probs

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
    with open("test.txt", encoding="UTF-8") as f:
        conversations = remove_sentences_with_numbers_and_symbols(f.read().lower().split(".")[:KB_MEMORY_UNCOMPRESSED])
    vocab = list(set(ngram for conv in conversations for ngram in create_ngrams_and_words(conv + ".", N)))
    vocab.append(PADDING_TOKEN)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    save_dict(word_to_idx, "langA.dat")
    save_dict(idx_to_word, "langB.dat")
    centers = np.linspace(-1, 1, len(word_to_idx))
    

elif _choice_ == "l":
    word_to_idx = load_dict("langA.dat")
    idx_to_word = load_dict("langB.dat")
    ngram_encoding_index = load_dict("model.dat")
    centers = np.linspace(-1, 1, len(word_to_idx))
while True:
    user_input = filter_text(input("You: "))
    response_begin = chat(ngram_encoding_index, user_input, word_to_idx, GENERATE_LENGTH, N)
    print(f"AI: {response_begin}\n")
