from pathlib import Path

PATH_MOVIES = "../../../data/movies/"
FILE_MOVIES_UNSEEN = "../../../data/movies/movies-unseen.txt"


def load_movies(path=PATH_MOVIES) -> tuple[list[str], list[int]]:
    """
    0 for negative
    1 for positive
    """
    texts = []
    labels = []
    # Read the positive files
    for file in (Path(PATH_MOVIES) / "pos").glob("*.txt"):
        with open(file) as f:
            # Remove \n for consistency with unseen
            texts.append(f.read().replace("\n", ""))
            labels.append(1)
    # Read the negative files
    for file in (Path(PATH_MOVIES) / "neg").glob("*.txt"):
        with open(file) as f:
            texts.append(f.read().replace("\n", ""))
            labels.append(0)
    return texts, labels


def load_movies_unseen(file=FILE_MOVIES_UNSEEN) -> list[str]:
    res = []
    with open(file) as f:
        for line in f.readlines():
            res.append(line.replace("\n", ""))
    return res


def write_prediction_movies(y_pred, file):
    with open(file, "w") as file:
        for y in y_pred:
            if y == 1:
                file.write("P\n")
            elif y == 0:
                file.write("N\n")
            else:
                file.write("ERROR")
                print("Error")
