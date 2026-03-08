import codecs
import re
import os
import string
from functools import partial
from collections import Counter

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
from nltk.corpus import stopwords

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def preprocess(text, stop_words=None, stemmer=None):
    """
    Transform text to remove unwanted bits
    """

    # Make lowercase
    text = text.lower()

    # Remove punctuation and whitespaces
    punctuation = string.punctuation + string.whitespace
    text = text.translate(str.maketrans(punctuation, " " * len(punctuation)))

    # Remove numbers
    text = re.sub(r"\d", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Tokenize
    tokens = text.split()

    # Remove stop words
    if stop_words is not None:
        tokens = [token for token in tokens if token not in stop_words]

    # Stem
    if stemmer is not None:
        tokens = [stemmer.stem(token) for token in tokens]

    text = " ".join(tokens)

    return text


def vectorize(
    texts,
    vectorizer,
    preprocessor=preprocess,
    language="english",
    ngram_range=(1, 1),
    remove_stopwords=True,
    stem=True,
):
    if stem:
        stemmer = SnowballStemmer(language)
    else:
        stemmer = None

    if remove_stopwords:
        stop_words = stopwords.words(language)
    else:
        stop_words = None

    vectorizer = vectorizer(
        preprocessor=partial(preprocessor, stemmer=stemmer, stop_words=stop_words),
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    return X, vocab


def plot_word_cloud(vocab, values, title=None):
    freq_dict = dict(zip(vocab, values))
    wc = WordCloud()
    wc.generate_from_frequencies(freq_dict)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_frequencies(vocab, values, top_k=100):
    pairs = sorted(zip(vocab, values), key=lambda x: x[1], reverse=True)[:top_k]
    words, vals = map(list, zip(*pairs))  # zip is like a transpose
    plt.bar(range(top_k), vals, alpha=0.7)
    plt.xticks(
        ticks=range(top_k), labels=words, rotation=90, size=np.maximum(100 / top_k, 5)
    )
    plt.title(f"Top {top_k} words frequency distribution")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()


def compute_odds_ratio(X, labels):

    pos_mask = np.array(labels) == 1

    pos_counts = X[pos_mask, :].sum(0).A1 + 1
    neg_counts = X[~pos_mask, :].sum(0).A1 + 1

    pos_probs = pos_counts / pos_counts.sum()
    neg_probs = neg_counts / neg_counts.sum()

    pos_odds = pos_probs / (1 - pos_probs)
    neg_odds = neg_probs / (1 - neg_probs)
    odds_ratio = pos_odds / neg_odds

    return odds_ratio
