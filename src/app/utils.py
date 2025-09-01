from enum import Enum
import json
import logging


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(levelname)s:     %(name)s: %(message)s")


class EnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.value
        return super().default(o)
