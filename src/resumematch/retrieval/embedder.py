"""Thin wrapper around sentence-transformers so the rest of the codebase
never imports sentence_transformers directly — swapping the embedding
model later means changing one file, not grepping the whole project.
"""

from functools import lru_cache


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None  # lazy-loaded; don't pay model-load cost at import time

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Input: list of strings (chunk texts, or a single query wrapped
            in a length-1 list).
        Output: list of embedding vectors, same order as input.
        """
        model = self._load()
        return model.encode(texts, normalize_embeddings=True).tolist()

    @property
    def max_tokens(self) -> int:
        """Longest input the model actually reads (512 for bge-small);
        anything past this is silently truncated by encode()."""
        return self._load().max_seq_length

    def count_tokens(self, text: str) -> int:
        """Length in the MODEL's own tokens (wordpieces, no specials) --
        the unit max_tokens is measured in. Whitespace word counts
        undercount this ~2x on resume text."""
        tokenizer = self._load().tokenizer
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
