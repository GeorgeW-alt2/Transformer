# Large Language Model v18.2 - entity
import numpy as np
import pickle
import re

# Model parameters
KB_memory_uncompressed = 1000 # KB access, -1 for unlimited
generate_length = 25
n = 3
sigma = 0.7  # Width of the Gaussian functions
epochs = 25
padding_token = '<unk>'
psych_threshold = 0.7
def create_ngrams_and_words(text, max_n):
    words = text.split()
    ngrams_and_words = words.copy()  # Start with single words
    for n in range(2, max_n + 1):
        ngrams = zip(*[words[i:] for i in range(n)])
        ngrams_and_words.extend([' '.join(ngram) for ngram in ngrams])
    return ngrams_and_words

def gaussian_rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c)**2 / (2 * s**2))

def encode_ngram(ngram, token_vector, word_to_idx, centers, sigma):
    if ngram in word_to_idx:
        idx = word_to_idx[ngram]
        return idx, gaussian_rbf(token_vector, centers[idx], sigma)
    else:
        idx = word_to_idx[padding_token]
        return idx, gaussian_rbf(token_vector, centers[idx], sigma)

def encode_sentence(sentence, word_to_idx, centers, sigma, max_n):
    encoded = np.zeros(len(word_to_idx))
    tokens = create_ngrams_and_words(sentence, max_n)

    token_vector = np.zeros(len(word_to_idx))
    for token in tokens:
        if token in word_to_idx:
            token_vector[word_to_idx[token]] = 1
        else:
            token_vector[word_to_idx[padding_token]] = 1

    for token in tokens:
        idx, rbf_value = encode_ngram(token, token_vector, word_to_idx, centers, sigma)
        encoded[idx] = rbf_value

    return encoded

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exps / np.sum(exps)

def text_to_vector(text, word_to_idx):
    vector = np.zeros(len(word_to_idx))
    tokens = create_ngrams_and_words(text, n)
    for token in tokens:
        if token in word_to_idx:
            vector[word_to_idx[token]] = 1
        else:
            vector[word_to_idx[padding_token]] = 1
    return vector

def chat(ngram_encoding_index, question, word_to_idx, generate_length, n):
    input_vector = text_to_vector(question, word_to_idx)
    
    max_similarity = 0
    best_match = None
    for stored_text in encountered_texts:
        stored_vector = text_to_vector(stored_text, word_to_idx)
        similarity = cosine_similarity(input_vector, stored_vector)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = stored_text
    
    if max_similarity > 0.5:  # Adjust the threshold as needed
        return f"Match found: {best_match}"
    
    output = []
    encoded = np.zeros(len(word_to_idx))

    ngrams = create_ngrams_and_words(question, n)
    for ngram in ngrams:
        if ngram in ngram_encoding_index:
            idx, rbf_value = ngram_encoding_index[ngram]
            encoded[idx] = rbf_value

    for i in range(generate_length):
        probabilities = softmax(encoded.flatten())
        rng = np.random.default_rng()
        predicted_idx = rng.choice(range(len(probabilities)), p=probabilities)
        ngram = idx_to_word.get(predicted_idx, padding_token)
        output.append(ngram)

        encoded = np.zeros(len(word_to_idx))
        next_input = ' '.join(output)
        ngrams = create_ngrams_and_words(next_input, n)
        for ngram in ngrams:
            if ngram in ngram_encoding_index:
                idx, rbf_value = ngram_encoding_index[ngram]
                encoded[idx] = rbf_value
    generated_response = ' '.join(output)
    return generated_response

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
    filtered_sentences = []
    for sentence in sentences:
        if re.match(r'^[A-Za-z\s,.]+$', sentence):
            filtered_sentences.append(sentence)
    return filtered_sentences

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def train_ngram_encoding_index(conversations, word_to_idx, centers, sigma, learning_rate=0.01):
    for epoch in range(epochs):
        print(f'Training epoch {epoch + 1}/{epochs}')
        for sentence in conversations:
            ngrams = create_ngrams_and_words(sentence, n)
            token_vector = np.zeros(len(word_to_idx))
            for ngram in ngrams:
                if ngram in word_to_idx:
                    token_vector[word_to_idx[ngram]] = 1
                else:
                    token_vector[word_to_idx[padding_token]] = 1

            for ngram in ngrams:
                if ngram in ngram_encoding_index:
                    idx, rbf_value = ngram_encoding_index[ngram]
                    target_vector = np.roll(token_vector, -1)
                    error = rbf_value - target_vector[idx]
                    centers[idx] -= learning_rate * error * (target_vector[idx] - centers[idx])
    save_dict(centers, "centers.dat")

_choice_ = input("\nSave new model/Load old model?[s/l]:").lower()

word_to_idx = {}
idx_to_word = {}
ngram_encoding_index = {}
if _choice_ == "s":
    with open("test.txt", encoding="UTF-8") as f:
        conversations = f.read().lower().split(".")[:KB_memory_uncompressed]
    conversations = remove_sentences_with_numbers_and_symbols(conversations)
    print("Memory size: ", len(conversations))
    
    vocab = set()
    for conv in conversations:
        ngrams = create_ngrams_and_words(conv + ".", n)
        for ngram in ngrams:
            vocab.add(ngram)

    vocab.add(padding_token)

    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    save_dict(word_to_idx, "langA.dat")
    save_dict(idx_to_word, "langB.dat")

    centers = np.linspace(-1, 1, len(word_to_idx))
    total_ngrams = len(vocab)
    current_progress = 0
    print_progress_bar(current_progress, total_ngrams, prefix='AutoGen:', suffix='Complete', length=50)
    for ngram in vocab:
        token_vector = np.zeros(len(word_to_idx))
        if ngram in word_to_idx:
            token_vector[word_to_idx[ngram]] = 1
        else:
            token_vector[word_to_idx[padding_token]] = 1
        idx, rbf_value = encode_ngram(ngram, token_vector, word_to_idx, centers, sigma)
        ngram_encoding_index[ngram] = (idx, rbf_value)
        current_progress += 1
        print_progress_bar(current_progress, total_ngrams, prefix='AutoGen:', suffix='Complete', length=50)
    
    save_dict(ngram_encoding_index, "model.dat")
    train_ngram_encoding_index(conversations, word_to_idx, centers, sigma)

if _choice_ == "l":
    word_to_idx = load_dict("langA.dat")
    idx_to_word = load_dict("langB.dat")
    ngram_encoding_index = load_dict("model.dat")
    centers = load_dict("centers.dat")

mind_aspects = [
    "Attention",
    "Memory",
    "Perception",
    "Cognition",
    "Consciousness",
    "Emotion",
    "Reasoning",
    "Imagination",
    "Learning",
    "Intuition",
    "Judgment",
    "Awareness",
    "Focus",
    "Creativity",
    "Problem-Solving",
    "Decision-Making",
    "Thinking",
    "Planning",
    "Language",
    "Self-Control",
    "Insight",
    "Empathy",
    "Mindfulness",
    "Self-Awareness",
    "Abstract Thinking",
    "Critical Thinking",
    "Analytical Thinking",
    "Creative Thinking",
    "Reflective Thinking",
    "Spatial Thinking",
    "Logical Thinking",
    "Emotional Intelligence",
    "Reasoning",
    "Perceptual Thinking",
    "Conceptual Thinking",
    "Decision-Making",
    "Problem-Solving",
    "Meta-Cognition",
    "Attention Span",
    "Working Memory",
    "Long-Term Memory",
    "Short-Term Memory",
    "Learning Styles",
    "Cognitive Biases",
    "Thinking Patterns",
    "Motivation",
    "Insightfulness",
    "Self-Efficacy",
    "Stress Management",
    "Cognitive Flexibility",
    "Mental Imagery"
]



while True:
    aspects = []
    encountered_texts = [] # add environment
    user_input = input("You: ")
    response_begin = chat(ngram_encoding_index, user_input, word_to_idx, generate_length, n)

    for aspect in mind_aspects:
        response = chat(ngram_encoding_index, aspect.lower(), word_to_idx, generate_length, n)
        aspects.append(response) 
    for i,aspect_unit in enumerate(aspects):
        X = encode_sentence(response_begin, word_to_idx, centers, sigma, n)
        Y = encode_sentence(aspect_unit, word_to_idx, centers, sigma, n)
        if cosine_similarity(X, Y) > psych_threshold:
            print("Mode:",mind_aspects[i])
            break
    print(f"AI: {response_begin}")
    print()
