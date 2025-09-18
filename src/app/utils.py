from enum import Enum
import json
import logging
import re

import nltk


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(levelname)s:     %(name)s: %(message)s")


class EnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        return super().default(o)


def _has_nltk_punkt() -> bool:
    try:
        nltk.data.find("tokenizers/punkt")
        return True
    except LookupError:
        try:
            nltk.data.find("tokenizers/punkt_tab")
            return True
        except LookupError:
            return False


def safe_sent_tokenize(text: str) -> list[str]:
    if _has_nltk_punkt():
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            pass
    # Fallback: simple regex-based sentence split
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", (text or "").strip())
    return [p for p in parts if p]
