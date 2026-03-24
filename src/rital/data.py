import re
from pathlib import Path
import numpy as np

# File paths for data

FILE_PRESIDENTS = "../../../data/presidents/presidents.utf8"
FILE_PRESIDENTS_UNSEEN = "../../../data/presidents/presidents-unseen.utf8"
PATH_MOVIES = "../../../data/movies/"
FILE_MOVIES_UNSEEN = "../../../data/movies/movies-unseen.txt"

# Functions for loading data
def load_presidents(file=FILE_PRESIDENTS) -> tuple[np.ndarray, np.ndarray]:
    """
    0 for Chirac
    1 for Mitterrand
    """
    texts = []
    labels = []
    with open(file) as f:
        for line in f.readlines():
            speaker, sentence = re.match(r"<\d+:\d+:(.)>\s*(.*)\n", line).groups()
            if speaker == "C":
                speaker = 0
            elif speaker == "M":
                speaker = 1
            else:
                # Something went wrong
                raise ValueError
            texts.append(sentence)
            labels.append(speaker)
    return np.array(texts), np.array(labels)


def load_presidents_unseen(file=FILE_PRESIDENTS_UNSEEN) -> list[str]:
    x = []
    with open(file) as f:
        for line in f.readlines():
            sentence = re.match(r"<\d+:\d+>\s*(.*)\n", line).group(1)
            x.append(sentence)
    return x
