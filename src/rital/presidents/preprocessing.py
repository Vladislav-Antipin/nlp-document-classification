import string
import re

LOWERCASE_EXCEPTIONS = ["monsieur", "madame", "messieurs", "mesdames", "état","président", "gouvernement", "maire", "ministre"]
# don't remove angles since they identify tags
PUNCTUATION_EXCEPTIONS = [r"<", r">", r"%"]

def preprocess(text, lowercase=True, remove_punctuation=True,stop_words=None, stemmer=None):
    """
    Transform text to remove unwanted bits
    """

    # Remove punctuation and whitespaces
    if remove_punctuation:
        punctuation = string.punctuation + string.whitespace
        for exception in PUNCTUATION_EXCEPTIONS:
            punctuation = punctuation.replace(exception, "")
        text = text.translate(str.maketrans(punctuation, " " * len(punctuation)))

    # Replace numbers by tokens
    text = re.sub(r"\d+", "<num>", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Tokenize
    tokens = text.split()
    # Make lowercase
    if lowercase:
        tokens = [ token.lower() if token.lower() not in LOWERCASE_EXCEPTIONS else token for token in tokens ]

    # Remove stop words
    if stop_words is not None:
        tokens = [token for token in tokens if token not in stop_words]

    # Stem
    if stemmer is not None:
        tokens = [stemmer.stem(token) for token in tokens]

    text = " ".join(tokens)

    return text

def tokenizer(text):
    return text.split()