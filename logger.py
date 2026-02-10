import json
import sys
import time
from datetime import datetime, timezone
from typing import Any


class StructuredLogger:
    def __init__(self, name: str):
        self.name = name

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
        }
        if kwargs:
            log_entry.update(kwargs)
        
        json.dump(log_entry, sys.stdout)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("WARNING", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("DEBUG", message, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)
