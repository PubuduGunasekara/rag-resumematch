"""Cleans model-written text before anyone sees it.

llama3.2:3b writes em dashes and the occasional emoji, which read as
machine-made in a product that should feel like a careful human report.
Applied to everything the LLM writes (requirement names, summaries).
Never applied to quotes from the resume or a posting: those are shown
exactly as the source has them, because they are evidence.
"""

import re

_RANGE_DASH = re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)")       # 2018–2021 -> 2018 to 2021
_CLAUSE_DASH = re.compile(r"\s*[—–]\s*|\s+-{1,2}\s+")         # "fast — and" / "fast -- and" -> "fast, and"
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF️‍⭐✅❌]"
)
_SPACES = re.compile(r"[ \t]{2,}")


def clean_generated(text: str) -> str:
    text = _EMOJI.sub("", text)
    text = _RANGE_DASH.sub(" to ", text)
    text = _CLAUSE_DASH.sub(", ", text)
    text = re.sub(r",\s*,", ",", text)          # "x, , y" from a dash next to a comma
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"^[,\s]+|[,\s]+$", "", text)  # a leading/trailing dash became a stray comma
    return _SPACES.sub(" ", text)
