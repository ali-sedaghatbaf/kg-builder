import json
import logging
from enum import Enum


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(levelname)s:     %(name)s: %(message)s")


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
