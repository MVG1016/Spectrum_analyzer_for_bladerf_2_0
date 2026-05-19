"""Tee stdout/stderr to a log file."""
import sys


class Logger:
    """Logger that writes to both console and file"""

    def __init__(self, filepath: str):
        self.terminal = sys.__stdout__
        self.log = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, message: str):
        if self.terminal:
            try:
                self.terminal.write(message)
            except Exception:
                pass
        try:
            self.log.write(message)
            self.log.flush()
        except Exception:
            pass

    def flush(self):
        if self.terminal:
            try:
                self.terminal.flush()
            except Exception:
                pass
        try:
            self.log.flush()
        except Exception:
            pass
