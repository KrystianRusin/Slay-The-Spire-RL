"""Calling a function periodically on a background thread."""

import logging
import threading

logger = logging.getLogger(__name__)


class Heartbeat:
    """Calls beat every interval seconds on a daemon thread while the with block runs.

    A beat that raises is logged, and the next one still runs on schedule.
    """

    def __init__(self, interval, beat, name):
        self.interval = interval
        self.name = name
        self._beat = beat
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{name}", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stopped.set()
        self._thread.join()
        return False

    def _run(self):
        while not self._stopped.wait(self.interval):
            try:
                self._beat()
            except Exception:
                logger.exception("Heartbeat %s failed", self.name)
