"""
preprocessor.py
---------------
Text preprocessing utilities:
  - cleaning (URLs, mentions, HTML)
  - tokenisation
  - stopword removal
  - lemmatisation
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# Download required NLTK data (runs once, silently)
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

_tokenizer  = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Keep some sentiment-bearing negations
_KEEP = {"no", "not", "nor", "never", "neither", "hardly", "barely", "scarcely"}
_STOP_FILTERED = _stop_words - _KEEP


def clean_text(text: str) -> str:
    """Remove URLs, HTML tags, punctuation, and numeric tokens."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # URLs
    text = re.sub(r"<.*?>", "", text)                       # HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)               # keep only alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize(text: str) -> list[str]:
    return _tokenizer.tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _STOP_FILTERED and len(t) > 2]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """Full pipeline → returns cleaned, lemmatised string ready for TF-IDF."""
    cleaned = clean_text(text)
    tokens  = tokenize(cleaned)
    tokens  = remove_stopwords(tokens)
    tokens  = lemmatize(tokens)
    return " ".join(tokens)
