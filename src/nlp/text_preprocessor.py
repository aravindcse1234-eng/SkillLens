import re
import nltk
from typing import List, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:
    def __init__(self):
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = str(text)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return nltk.word_tokenize(self.clean_text(text))

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [t for t in tokens if t.lower() not in self.stop_words and len(t) > 2]

    def preprocess_for_ner(self, text: str) -> str:
        text = self.clean_text(text)
        text = " ".join(self.remove_stopwords(self.tokenize(text)))
        return text

    def split_into_chunks(self, text: str, max_length: int = 512) -> List[str]:
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > max_length:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    def extract_sentences(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(self.clean_text(text))
        return [s.strip() for s in sentences if len(s.strip()) > 10]
