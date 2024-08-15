import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from nltk.util import ngrams
from nltk import FreqDist
import numpy as np
import textstat
import pandas as pd

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize tokenizer and model for perplexity calculation
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

def syntactic_complexity(text):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    total_sentences = len(sentences)
    total_tokens = sum(len(token) for token in tokens)
    avg_sentence_length = total_tokens / total_sentences if total_sentences > 0 else 0
    return avg_sentence_length

def lexical_richness(text):
    tokens = word_tokenize(text)
    types = set(tokens)
    ttr = len(types) / len(tokens) if len(tokens) > 0 else 0
    return ttr

def semantic_coherence(text, num_topics=5):
    tokens = [word_tokenize(text.lower())]
    if not tokens or all(not token for token in tokens):
        return np.nan
    dictionary = Dictionary(tokens)
    if len(dictionary) == 0:
        return np.nan
    corpus = [dictionary.doc2bow(token) for token in tokens]
    if not corpus:
        return np.nan
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    top_topics = lda.top_topics(corpus)
    coherence_score = sum(score for topic, score in top_topics) / num_topics if top_topics else np.nan
    return coherence_score

def sentiment_score(text):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(text)
    return score['compound']

def calculate_burstiness(text):
    words = word_tokenize(text.lower())
    freq_dist = FreqDist(words)
    frequencies = list(freq_dist.values())
    mean_freq = np.mean(frequencies)
    std_dev_freq = np.std(frequencies)
    burstiness = std_dev_freq / mean_freq if mean_freq > 0 else 0
    return burstiness

def calculate_perplexity(text):
    tokenize_input = tokenizer_gpt2.encode(text, add_special_tokens=True, max_length=1024, truncation=True)
    tensor_input = torch.tensor([tokenize_input]).to(model_gpt2.device)
    with torch.no_grad():
        outputs = model_gpt2(tensor_input, labels=tensor_input)
        loss = outputs.loss
    return np.exp(loss.item()) if loss is not None else np.nan

def generate_ngram_features(text, n=2):
    tokens = word_tokenize(text)
    ngrams_list = list(ngrams(tokens, n))
    return len(set(ngrams_list))

def compute_features(text):
    features = {}
    features['readability_score'] = textstat.flesch_reading_ease(text)
    features['syntactic_complexity'] = syntactic_complexity(text)
    features['lexical_richness'] = lexical_richness(text)
    features['sentiment_score'] = sentiment_score(text)
    features['burstiness'] = calculate_burstiness(text)
    features['semantic_coherence'] = semantic_coherence(text)
    features['perplexity'] = calculate_perplexity(text)
    features['unique_trigrams'] = generate_ngram_features(text, 3)
    return pd.DataFrame([features])
