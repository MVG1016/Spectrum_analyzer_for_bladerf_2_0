"""Background TX worker that loops sync_tx and surfaces stream errors."""
import time

from PyQt5 import QtCore
from PyQt5.QtCore import QThread


class TXThread(QThread):
    """Continuous TX transmission thread.

    Emits ``error`` (str, error_count) the first time sync_tx raises, and
    again every ERROR_REPORT_INTERVAL failures so the UI gets a heartbeat
    without being flooded.
    """

    error = QtCore.pyqtSignal(str, int)
    ERROR_REPORT_INTERVAL = 100

    def __init__(self, sdr, tx_buffer, parent=None):
        super().__init__(parent)
        self.sdr = sdr
        self.tx_buffer = tx_buffer
        self.running = True
        self.error_count = 0

    def run(self):
        """Continuous transmission loop"""
        while self.running:
            try:
                self.sdr.sync_tx(self.tx_buffer, len(self.tx_buffer) // 4)
                time.sleep(0.001)
            except Exception as e:
                self.error_count += 1
                if self.error_count == 1 or \
                   self.error_count % self.ERROR_REPORT_INTERVAL == 0:
                    print(f"TX send error #{self.error_count}: {e}")
                    self.error.emit(str(e), self.error_count)
                time.sleep(0.01)

    def stop(self):
        """Stop transmission"""
        self.running = False
        self.wait()
