import re


def normalize(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # remove html tags
    text = text.lower().strip()  # lowercase for consistency
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)  # remove spaces before .,!?;:
    text = re.sub(r"\(\s+", "(", text)  # remove spaces after (
    text = re.sub(r"\s+\)", ")", text)  # remove spaces before )
    return text
