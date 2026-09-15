"""Structured logging: every record in a process written to stderr as one JSON object per line."""

import json
import logging
import os
from datetime import datetime, timezone

LEVEL_VAR = "LOG_LEVEL"


class JsonFormatter(logging.Formatter):
    """Formats a record as a JSON object with its time, level, logger and message, plus fixed context fields such as actor_id."""

    def __init__(self, **context):
        super().__init__()
        self._context = context

    def format(self, record):
        entry = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self._context,
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(**context):
    """Send every log record in this process to stderr as JSON carrying context, at the level named by LOG_LEVEL (INFO by default)."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(**context))
    logging.basicConfig(level=os.environ.get(LEVEL_VAR, "INFO").upper(), handlers=[handler], force=True)
