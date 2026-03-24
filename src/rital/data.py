import re
from pathlib import Path

# File paths for data
FILE_PRESIDENTS = "../../data/presidents/presidents.utf8"
FILE_PRESIDENTS_UNSEEN = "../../data/presidents/presidents-unseen.utf8"

# Functions for loading data
def load_presidents(file=FILE_PRESIDENTS) -> tuple[list[str], list[int]]:
    """
    0 for Mitterrand
    1 for Chirac
    """
    texts = []
    labels = []
    with open(file) as f:
        for line in f.readlines():
            speaker, sentence = re.match(r"<\d+:\d+:(.)>\s*(.*)\n", line).groups()
            if speaker == "M":
                speaker = 0
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
