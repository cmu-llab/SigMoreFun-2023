import re

word_regex = r"[^.,!?;\s]+|[.,!?;]"


def word_tokenize(s: str):
    """Tokenizes by splitting into spaces, leaving punctuation as tokens"""
    return re.findall(word_regex, s)


word_regex_no_punc = (
    r"[^.,!?;\s]+"  # Matches any string not containing punctuation or whitespace
)


def word_tokenize_no_punc(s: str):
    """Tokenizes by splitting into spaces, skipping punctuation"""
    return re.findall(word_regex_no_punc, s)


# Matches any string not containing punctuation or whitespace, and splits before "-"
# Added punctuation from Lezgi -LT
# morpheme_regex_no_punc = r"-?[^.,!?;«»\s-]+"
morpheme_regex_no_punc = r"-?[^.,!?;\s-]+"


def morpheme_tokenize_no_punc(s: str):
    """Tokenizes by splitting into morphemes, skipping punctuation"""
    return re.findall(morpheme_regex_no_punc, s)


morpheme_regex = r"-?[^.,!?;«»\s-]+|[.,!?;«»]+"


def morpheme_tokenize(s: str):
    """Tokenizes by splitting into morphemes, preserving punctuation"""
    return re.findall(morpheme_regex, s)


def gloss_tokenize(s: str):
    """Tokenizes by splitting into morphemes, preserving punctuation"""
    return re.split(r"\s|-", s)


tokenizers = {
    "word": word_tokenize,
    "word_no_punc": word_tokenize_no_punc,
    "morpheme": morpheme_tokenize,
    "morpheme_no_punc": morpheme_tokenize_no_punc,
    "gloss": gloss_tokenize,
}
