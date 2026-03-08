import re
from pathlib import Path

# File paths for data
FILE_PRESIDENTS = "../../data/presidents/presidents.utf8"
FILE_PRESIDENTS_UNSEEN = "../../data/presidents/presidents-unseen.utf8"
PATH_MOVIES = "../../data/movies/"
FILE_MOVIES_UNSEEN = "../../data/movies/movies-unseen.txt"


# Functions for loading data
def load_presidents(file=FILE_PRESIDENTS) -> tuple[list[str], list[int]]:
    """
    -1 for Mitterrand
    +1 for Chirac
    """
    texts = []
    labels = []
    with open(file) as f:
        for line in f.readlines():
            speaker, sentence = re.match(r"<\d+:\d+:(.)>\s*(.*)\n", line).groups()
            if speaker == "M":
                speaker = -1
            elif speaker == "C":
                speaker = 1
            else:
                # Something went wrong
                raise ValueError
            texts.append(sentence)
            labels.append(speaker)
    return texts, labels


def load_presidents_unseen(file=FILE_PRESIDENTS_UNSEEN) -> list[str]:
    x = []
    with open(file) as f:
        for line in f.readlines():
            sentence = re.match(r"<\d+:\d+>\s*(.*)\n", line).group(1)
            x.append(sentence)
    return x


def load_movies(path=PATH_MOVIES) -> tuple[list[str], list[int]]:
    """
    -1 for negative
    +1 for positive
    """
    texts = []
    labels = []
    # Read the positive files
    for file in (Path(PATH_MOVIES) / "pos").glob("*.txt"):
        with open(file) as f:
            texts.append("".join(f.readlines()))
            labels.append(1)
    # Read the negative files
    for file in (Path(PATH_MOVIES) / "neg").glob("*.txt"):
        with open(file) as f:
            texts.append("".join(f.readlines()))
            labels.append(-1)
    return texts, labels


def load_movies_unseen(file=FILE_MOVIES_UNSEEN) -> list[str]:
    with open(file) as f:
        return f.readlines()
