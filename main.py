"""
BladeRF 2.0 Spectrum Analyzer and Signal Generator
Refactored version with improved architecture and error handling
"""

import numpy as np
import time
import sys
import os
from datetime import datetime
from threading import Thread, Lock
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

from bladerf import _bladerf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                            QPushButton, QHBoxLayout, QComboBox, QTabWidget,
                            QLineEdit, QLabel, QMessageBox)
from PyQt5.QtCore import QTimer, QLocale, QObject, pyqtSignal, QThread
from PyQt5.QtGui import QDoubleValidator, QIcon
from PyQt5 import QtCore
import pyqtgraph as pg


# ============================================================================
# Configuration and Data Classes
# ============================================================================

@dataclass
class SDRConfig:
    """SDR configuration parameters"""
    # Frequency parameters (Hz)
    start_freq: float = 80e6
    end_freq: float = 120e6
    step: float = 0.5e6
    span: float = 10e6

    # Sampling parameters
    sample_rate: float = 10e6
    num_samples: int = 32768
    gain: int = 18

    # Window type for FFT
    window_type: str = 'hann'

    # BladeRF streaming configuration
    SYNC_NUM_BUFFERS: int = 16
    SYNC_NUM_TRANSFERS: int = 8
    SYNC_STREAM_TIMEOUT: int = 3500
    BUFFER_SIZE_MULTIPLIER: int = 4  # SC16_Q11: 2 bytes/sample * 2 (I+Q)

    # Waterfall configuration
    waterfall_steps: int = 20

    def validate(self) -> Tuple[bool, str]:
        """Validate configuration parameters"""
        if self.start_freq >= self.end_freq:
            return False, "Start frequency must be less than end frequency"

        if self.step <= 0 or self.step > (self.end_freq - self.start_freq):
            return False, "Invalid step size"

        if self.span <= 0 or self.span > 100e6:
            return False, "Span must be between 0 and 100 MHz"

        if self.sample_rate <= 0 or self.sample_rate > 100e6:
            return False, "Sample rate must be between 0 and 100 MHz"

        if self.num_samples < 128 or self.num_samples > 1_000_000:
            return False, "Number of samples must be between 128 and 1,000,000"

        if self.gain < 0 or self.gain > 60:
            return False, "Gain must be between 0 and 60 dB"

        return True, ""


@dataclass
class CalibrationData:
    """Calibration data for a specific frequency"""
    freq_offset: float = 0.0
    power_offset: float = 0.0


# ============================================================================
# Signal Processing
# ============================================================================

class SignalProcessor:
    """Signal processing utilities for spectrum analysis"""

    @staticmethod
    def create_window(window_type: str, num_samples: int) -> np.ndarray:
        """Create window function"""
        if window_type == 'hann':
            return np.hanning(num_samples)
        elif window_type == 'hamming':
            return np.hamming(num_samples)
        elif window_type == 'blackman':
            return np.blackman(num_samples)
        else:
            return np.ones(num_samples)

    @staticmethod
    def calculate_spectrum(samples: np.ndarray, window: np.ndarray) -> np.ndarray:
        """Calculate power spectrum in dBm"""
        # Apply window
        windowed_samples = samples * window

        # Calculate window correction factor
        window_correction = np.sum(window ** 2) / len(window)

        # FFT
        spectrum = np.fft.fftshift(np.fft.fft(windowed_samples))

        # Power calculation with proper normalization
        power = (np.abs(spectrum) ** 2) / (len(samples) ** 2 * window_correction)

        # Convert to dBm (avoiding log of zero)
        power_db = 10 * np.log10(np.maximum(power, 1e-20))

        return power_db

    @staticmethod
    def apply_calibration(spectrum: np.ndarray, cal_data: CalibrationData) -> np.ndarray:
        """Apply calibration correction to spectrum"""
        return spectrum + cal_data.power_offset


# ============================================================================
# SDR Controller
# ============================================================================

class SDRController:
    """BladeRF device controller with thread-safe operations"""

    def __init__(self):
        self.sdr = _bladerf.BladeRF()
        self.rx: Optional[_bladerf.Channel] = None
        self.tx: Optional[_bladerf.Channel] = None
        self.lock = Lock()

        # Calibration table
        self.calibration_table: Dict[float, CalibrationData] = {
            100e6:  CalibrationData(),
            500e6:  CalibrationData(),
            1000e6: CalibrationData(),
            2000e6: CalibrationData(),
            2400e6: CalibrationData(),
            3000e6: CalibrationData(),
            4000e6: CalibrationData(),
            5000e6: CalibrationData(),
            5800e6: CalibrationData(),
        }

    def get_calibration(self, freq: float) -> CalibrationData:
        """Get calibration data for given frequency (closest match)"""
        if not self.calibration_table:
            return CalibrationData()

        closest_freq = min(self.calibration_table.keys(),
                          key=lambda f: abs(f - freq))
        return self.calibration_table[closest_freq]

    def configure_rx_channel(self, channel_index: int, config: SDRConfig):
        """Configure RX channel with thread safety"""
        with self.lock:
            try:
                # Disable previous channel
                if self.rx is not None:
                    try:
                        self.rx.enable = False
                    except Exception as e:
                        print(f"Warning: Error disabling RX: {e}")
                    time.sleep(0.2)  # Allow hardware to settle

                # Create new channel
                self.rx = self.sdr.Channel(_bladerf.CHANNEL_RX(channel_index))

                # Configure parameters
                self.rx.sample_rate = int(config.sample_rate)
                self.rx.bandwidth = int(config.span)
                self.rx.gain_mode = _bladerf.GainMode.Manual
                self.rx.gain = config.gain

                # Configure streaming BEFORE enabling
                self.sdr.sync_config(
                    layout=_bladerf.ChannelLayout.RX_X1,
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=config.SYNC_NUM_BUFFERS,
                    buffer_size=config.num_samples * config.BUFFER_SIZE_MULTIPLIER,
                    num_transfers=config.SYNC_NUM_TRANSFERS,
                    stream_timeout=config.SYNC_STREAM_TIMEOUT
                )

                # Enable channel after configuration
                self.rx.enable = True
                time.sleep(0.1)  # Allow hardware to settle

                print(f"RX channel {channel_index} configured successfully")

            except _bladerf.BladeRFError as e:
                print(f"BladeRF error configuring RX channel: {e}")
                raise
            except Exception as e:
                print(f"Error configuring RX channel: {e}")
                import traceback
                traceback.print_exc()
                raise

    def configure_tx_channel(self, channel_index: int, config: SDRConfig):
        """Configure TX channel with thread safety"""
        with self.lock:
            try:
                # Disable previous channel
                if self.tx is not None:
                    try:
                        self.tx.enable = False
                    except Exception as e:
                        print(f"Warning: Error disabling TX: {e}")
                    time.sleep(0.2)

                # Create new channel
                self.tx = self.sdr.Channel(_bladerf.CHANNEL_TX(channel_index))

                # Configure parameters
                self.tx.sample_rate = int(config.sample_rate)
                self.tx.bandwidth = int(config.span)
                self.tx.gain_mode = _bladerf.GainMode.Manual
                self.tx.gain = config.gain

                # Configure streaming BEFORE enabling
                self.sdr.sync_config(
                    layout=_bladerf.ChannelLayout.TX_X1,
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=config.SYNC_NUM_BUFFERS,
                    buffer_size=config.num_samples * config.BUFFER_SIZE_MULTIPLIER,
                    num_transfers=config.SYNC_NUM_TRANSFERS,
                    stream_timeout=config.SYNC_STREAM_TIMEOUT
                )

                # Enable channel after configuration
                self.tx.enable = True
                time.sleep(0.1)

                print(f"TX channel {channel_index} configured successfully")

            except _bladerf.BladeRFError as e:
                print(f"BladeRF error configuring TX channel: {e}")
                raise
            except Exception as e:
                print(f"Error configuring TX channel: {e}")
                import traceback
                traceback.print_exc()
                raise

    def read_samples(self, freq: float, num_samples: int) -> np.ndarray:
        """Read samples at specified frequency"""
        with self.lock:
            try:
                if self.rx is None:
                    raise RuntimeError("RX channel not initialized")

                # Set frequency
                self.rx.frequency = int(freq)
                time.sleep(0.0003)  # Increased settling time for stability

                # Allocate buffer
                buf = bytearray(num_samples * 4)

                # Receive samples with timeout handling
                try:
                    self.sdr.sync_rx(buf, num_samples)
                except _bladerf.BladeRFError as e:
                    if "timeout" in str(e).lower():
                        print(f"Timeout reading samples at {freq/1e6:.2f} MHz, retrying...")
                        time.sleep(0.01)
                        buf = bytearray(num_samples * 4)
                        self.sdr.sync_rx(buf, num_samples)
                    else:
                        raise

                # Convert to complex samples
                samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                samples = samples.view(np.complex64) / (2 ** 11)

                # Clear buffer to help with memory management
                del buf

                return samples

            except _bladerf.BladeRFError as e:
                print(f"BladeRF error reading samples at {freq/1e6:.2f} MHz: {e}")
                raise
            except Exception as e:
                print(f"Unexpected error reading samples: {e}")
                import traceback
                traceback.print_exc()
                raise

    def transmit_signal(self, freq: float, power: float, signal: np.ndarray):
        """Transmit signal at specified frequency and power"""
        with self.lock:
            try:
                if self.tx is None:
                    raise RuntimeError("TX channel not initialized")

                # Set parameters
                self.tx.frequency = int(freq)
                self.tx.gain = int(round(power))

                # Ensure TX is enabled
                self.tx.enable = True
                time.sleep(0.001)  # Small delay for frequency settling

                # Convert to SC16 Q11 format
                iq = np.empty(2 * len(signal), dtype=np.int16)
                iq[0::2] = np.clip(np.real(signal) * (2 ** 11), -2048, 2047).astype(np.int16)
                iq[1::2] = np.clip(np.imag(signal) * (2 ** 11), -2048, 2047).astype(np.int16)

                # Transmit
                self.sdr.sync_tx(iq.tobytes(), len(signal))

            except _bladerf.BladeRFError as e:
                print(f"BladeRF error transmitting signal at {freq/1e6:.2f} MHz: {e}")
                raise
            except Exception as e:
                print(f"Error transmitting signal: {e}")
                import traceback
                traceback.print_exc()
                raise

    def close(self):
        """Safely close SDR device"""
        with self.lock:
            try:
                if self.rx is not None:
                    self.rx.enable = False
                if self.tx is not None:
                    self.tx.enable = False
                self.sdr.close()
                print("SDR closed successfully")
            except Exception as e:
                print(f"Error closing SDR: {e}")


# ============================================================================
# Worker Threads
# ============================================================================

class ScanWorker(QObject):
    """Worker for spectrum scanning in separate thread"""

    finished = pyqtSignal()
    progress = pyqtSignal(int)
    result = pyqtSignal(np.ndarray, np.ndarray)  # freqs, spectrum
    error = pyqtSignal(str)

    def __init__(self, sdr: SDRController, config: SDRConfig):
        super().__init__()
        self.sdr = sdr
        self.config = config
        self.processor = SignalProcessor()
        self.running = True
        self._lock = Lock()

    def stop(self):
        """Stop the scan worker"""
        with self._lock:
            self.running = False

    def is_running(self):
        """Check if worker is still running"""
        with self._lock:
            return self.running

    def run(self):
        """Main scan loop"""
        try:
            # Create window function
            window = self.processor.create_window(
                self.config.window_type,
                self.config.num_samples
            )

            # Calculate center frequencies
            center_freqs = np.arange(
                self.config.start_freq,
                self.config.end_freq,
                self.config.step
            )

            # Initialize arrays
            full_spectrum = np.zeros(len(center_freqs) * self.config.num_samples)
            full_freqs = np.zeros(len(center_freqs) * self.config.num_samples)

            # Scan loop
            scan_start = time.time()

            for i, freq in enumerate(center_freqs):
                if not self.is_running():
                    print("Scan stopped by user")
                    break

                try:
                    # Get calibration
                    cal = self.sdr.get_calibration(freq)

                    # Read samples
                    samples = self.sdr.read_samples(
                        freq + cal.freq_offset,
                        self.config.num_samples
                    )

                    # Calculate spectrum
                    power_db = self.processor.calculate_spectrum(samples, window)
                    power_db = self.processor.apply_calibration(power_db, cal)

                    # Store results
                    start_idx = i * self.config.num_samples
                    end_idx = (i + 1) * self.config.num_samples

                    full_freqs[start_idx:end_idx] = np.fft.fftshift(
                        np.fft.fftfreq(len(samples), d=1 / self.config.sample_rate)
                    ) + freq

                    full_spectrum[start_idx:end_idx] = power_db

                    # Report progress
                    progress = int((i + 1) / len(center_freqs) * 100)
                    self.progress.emit(progress)

                except Exception as e:
                    print(f"Error scanning frequency {freq/1e6:.2f} MHz: {e}")
                    # Continue with next frequency instead of aborting
                    continue

            # Emit results only if scan completed
            if self.is_running():
                self.result.emit(full_freqs, full_spectrum)
                scan_time = time.time() - scan_start
                print(f"Scan completed in {scan_time:.2f}s")

        except Exception as e:
            error_msg = f"Scan error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)

        finally:
            with self._lock:
                self.running = False
            self.finished.emit()


class SweepWorker(QObject):
    """Worker for frequency sweep transmission"""

    finished = pyqtSignal()
    status_update = pyqtSignal(str, float)  # message, frequency
    error = pyqtSignal(str)

    def __init__(self, sdr: SDRController, config: SDRConfig,
                 start_freq: float, stop_freq: float, step_freq: float,
                 delay_ms: int, power: float):
        super().__init__()
        self.sdr = sdr
        self.config = config
        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.step_freq = step_freq
        self.delay_ms = delay_ms
        self.power = power
        self.running = True
        self._lock = Lock()

    def stop(self):
        """Stop the sweep"""
        with self._lock:
            self.running = False

    def is_running(self):
        """Check if worker is still running"""
        with self._lock:
            return self.running

    def run(self):
        """Main sweep loop"""
        try:
            # Ensure TX is configured with sync
            with self.sdr.lock:
                if self.sdr.tx is None:
                    raise RuntimeError("TX channel not initialized")

                # Reconfigure sync for TX to ensure it's initialized
                print("Configuring TX sync...")
                self.sdr.sdr.sync_config(
                    layout=_bladerf.ChannelLayout.TX_X1,
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=self.config.SYNC_NUM_BUFFERS,
                    buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                    num_transfers=self.config.SYNC_NUM_TRANSFERS,
                    stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                )

                # Ensure TX is enabled
                self.sdr.tx.enable = True
                time.sleep(0.1)

            # Generate test signal (1 MHz tone)
            t = np.arange(self.config.num_samples) / self.config.sample_rate
            tx_signal = 0.7 * np.exp(2j * np.pi * 1e6 * t)

            print("Starting sweep transmission...")

            # Sweep loop
            while self.is_running():
                for freq in np.arange(self.start_freq,
                                     self.stop_freq + self.step_freq,
                                     self.step_freq):
                    if not self.is_running():
                        break

                    try:
                        # Transmit at current frequency
                        self.sdr.transmit_signal(freq, self.power, tx_signal)

                        # Update status
                        self.status_update.emit(
                            f"Sweep TX @ {freq/1e6:.2f} MHz",
                            freq
                        )

                        # Delay
                        time.sleep(self.delay_ms / 1000.0)

                    except Exception as e:
                        print(f"Error transmitting at {freq/1e6:.2f} MHz: {e}")
                        # Continue with next frequency
                        continue

            # Disable TX when done
            with self.sdr.lock:
                if self.sdr.tx is not None:
                    self.sdr.tx.enable = False

            print("Sweep transmission stopped")

        except _bladerf.BladeRFError as e:
            error_msg = f"BladeRF sweep error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)
        except Exception as e:
            error_msg = f"Sweep error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error.emit(error_msg)

        finally:
            with self._lock:
                self.running = False
            self.finished.emit()


# ============================================================================
# Main Window
# ============================================================================

class SpectrumAnalyzer(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Configuration
        self.config = SDRConfig()

        # SDR Controller
        self.sdr = SDRController()

        # Workers
        self.scan_worker: Optional[ScanWorker] = None
        self.scan_thread: Optional[QThread] = None
        self.sweep_worker: Optional[SweepWorker] = None
        self.sweep_thread: Optional[QThread] = None

        # Scanning state
        self.is_scanning = False
        self.scan_lock = Lock()

        # Data storage
        self.full_freqs = np.array([])
        self.full_spectrum = np.array([])
        self.max_hold_spectrum = np.array([])
        self.max_hold_enabled = False

        # Waterfall data
        self.waterfall_data = np.zeros((
            self.config.waterfall_steps,
            len(np.arange(self.config.start_freq, self.config.end_freq,
                         self.config.step)) * self.config.num_samples
        ))
        self.waterfall_ptr = 0

        # Initialize GUI
        self.init_ui()

        # Initialize SDR
        try:
            self.sdr.configure_rx_channel(0, self.config)
            self.sdr.configure_tx_channel(0, self.config)
        except Exception as e:
            QMessageBox.critical(self, "SDR Error",
                               f"Failed to initialize SDR: {str(e)}")

        # Update timer for continuous scanning
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.start_scan)

    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("BladeRF 2.0 Spectrum Analyzer and Signal Generator")
        self.setGeometry(100, 100, 1200, 800)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.rx_tab = QWidget()
        self.tx_tab = QWidget()
        self.tab_widget.addTab(self.rx_tab, "Receive Channel")
        self.tab_widget.addTab(self.tx_tab, "Transmit Channel")

        # Initialize tab contents
        self.init_rx_ui()
        self.init_tx_ui()

    def init_rx_ui(self):
        """Initialize RX tab UI"""
        layout = QVBoxLayout()

        # ===== Spectrum Plot =====
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setLabel('left', 'Power (dBm)')
        self.graphWidget.setLabel('bottom', 'Frequency (MHz)')
        self.graphWidget.setYRange(-120, 0)
        self.graphWidget.showGrid(x=True, y=True)

        self.spectrum_curve = self.graphWidget.plot(pen='y')
        self.max_hold_curve = self.graphWidget.plot(
            pen=pg.mkPen('c', style=QtCore.Qt.DashLine)
        )

        # Peak marker
        self.max_marker = pg.ScatterPlotItem(
            size=15,
            pen=pg.mkPen('r'),
            brush=pg.mkBrush('r')
        )
        self.graphWidget.addItem(self.max_marker)

        self.max_text = pg.TextItem(anchor=(0.5, 1.5), color='w', fill='k')
        self.graphWidget.addItem(self.max_text)

        # ===== Waterfall Plot =====
        self.waterfallWidget = pg.PlotWidget()
        self.waterfallImg = pg.ImageItem()
        self.waterfallWidget.addItem(self.waterfallImg)
        self.waterfallWidget.getViewBox().invertY(True)
        self.waterfallWidget.setLabel('left', 'Step')
        self.waterfallWidget.setLabel('bottom', 'Frequency (MHz)')

        # Colormap
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array([
            [0, 0, 255, 255],
            [0, 255, 255, 255],
            [0, 255, 0, 255],
            [255, 255, 0, 255],
            [255, 0, 0, 255]
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.waterfallImg.setLookupTable(cmap.getLookupTable())

        # ===== Parameters =====
        params_layout = QHBoxLayout()

        self.start_freq_input = QLineEdit(str(self.config.start_freq / 1e6))
        self.start_freq_input.setValidator(QDoubleValidator(1, 6000, 2))
        self.start_freq_input.setFixedWidth(80)

        self.end_freq_input = QLineEdit(str(self.config.end_freq / 1e6))
        self.end_freq_input.setValidator(QDoubleValidator(1, 6000, 2))
        self.end_freq_input.setFixedWidth(80)

        self.step_input = QLineEdit(str(self.config.step / 1e6))
        self.step_input.setValidator(QDoubleValidator(0.1, 100, 2))
        self.step_input.setFixedWidth(80)

        self.span_input = QLineEdit(str(self.config.span / 1e6))
        self.span_input.setValidator(QDoubleValidator(1, 100, 2))
        self.span_input.setFixedWidth(80)

        self.sample_rate_input = QLineEdit(str(self.config.sample_rate / 1e6))
        self.sample_rate_input.setValidator(QDoubleValidator(1, 100, 2))
        self.sample_rate_input.setFixedWidth(80)

        self.num_samples_input = QLineEdit(str(self.config.num_samples))
        self.num_samples_input.setValidator(QDoubleValidator(128, 1_000_000, 0))
        self.num_samples_input.setFixedWidth(80)

        self.gain_input = QLineEdit(str(self.config.gain))
        self.gain_input.setValidator(QDoubleValidator(0, 60, 0))
        self.gain_input.setFixedWidth(80)

        self.apply_rx_settings_btn = QPushButton("Apply RX Settings")
        self.apply_rx_settings_btn.clicked.connect(self.apply_rx_settings)

        params_layout.addWidget(QLabel("Start (MHz):"))
        params_layout.addWidget(self.start_freq_input)
        params_layout.addWidget(QLabel("Stop (MHz):"))
        params_layout.addWidget(self.end_freq_input)
        params_layout.addWidget(QLabel("Step (MHz):"))
        params_layout.addWidget(self.step_input)
        params_layout.addWidget(QLabel("Span (MHz):"))
        params_layout.addWidget(self.span_input)
        params_layout.addWidget(QLabel("Rate (MHz):"))
        params_layout.addWidget(self.sample_rate_input)
        params_layout.addWidget(QLabel("Samples:"))
        params_layout.addWidget(self.num_samples_input)
        params_layout.addWidget(QLabel("Gain:"))
        params_layout.addWidget(self.gain_input)
        params_layout.addWidget(self.apply_rx_settings_btn)

        # ===== Control Buttons =====
        buttons_layout = QHBoxLayout()

        self.scan_btn = QPushButton("Start Scan")
        self.scan_btn.setCheckable(True)
        self.scan_btn.clicked.connect(self.toggle_scan)

        self.toggle_maxhold_btn = QPushButton("Max Hold: OFF")
        self.toggle_maxhold_btn.setCheckable(True)
        self.toggle_maxhold_btn.clicked.connect(self.toggle_max_hold)

        self.rx_selector = QComboBox()
        self.rx_selector.addItems(["RX1", "RX2"])
        self.rx_selector.currentIndexChanged.connect(self.switch_rx_channel)

        buttons_layout.addWidget(self.scan_btn)
        buttons_layout.addWidget(self.toggle_maxhold_btn)
        buttons_layout.addWidget(QLabel("Channel:"))
        buttons_layout.addWidget(self.rx_selector)

        # ===== Layout Assembly =====
        layout.addLayout(params_layout)
        layout.addWidget(self.graphWidget)
        layout.addWidget(self.waterfallWidget)
        layout.addLayout(buttons_layout)

        self.rx_tab.setLayout(layout)

    def init_tx_ui(self):
        """Initialize TX tab UI"""
        layout = QVBoxLayout()

        # ===== Single Tone Transmission =====
        layout.addWidget(QLabel("<b>Single Tone Transmission</b>"))

        self.tx_freq_label = QLabel("Frequency: 5000 MHz")
        self.tx_freq_input = QLineEdit("5000")
        self.tx_freq_input.setValidator(QDoubleValidator(0, 6000, 2))
        self.tx_freq_input.setFixedWidth(100)

        self.tx_power_label = QLabel("Power: 10 dB")
        self.tx_power_input = QLineEdit("10")
        self.tx_power_input.setValidator(QDoubleValidator(0, 30, 1))
        self.tx_power_input.setFixedWidth(100)

        self.tx_selector = QComboBox()
        self.tx_selector.addItems(["TX1", "TX2"])
        self.tx_selector.currentIndexChanged.connect(self.switch_tx_channel)
        self.tx_selector.setFixedWidth(100)

        self.start_tx_btn = QPushButton("Start Transmission")
        self.start_tx_btn.setCheckable(True)
        self.start_tx_btn.clicked.connect(self.toggle_transmission)
        self.start_tx_btn.setFixedWidth(200)

        layout.addWidget(self.tx_freq_label)
        layout.addWidget(QLabel("Frequency (MHz):"))
        layout.addWidget(self.tx_freq_input)
        layout.addWidget(self.tx_power_label)
        layout.addWidget(QLabel("Power (dB):"))
        layout.addWidget(self.tx_power_input)
        layout.addWidget(QLabel("Channel:"))
        layout.addWidget(self.tx_selector)
        layout.addWidget(self.start_tx_btn)

        # ===== Sweep Mode =====
        layout.addWidget(QLabel("<b>Sweep Mode</b>"))

        self.sweep_start_freq = QLineEdit("5000")
        self.sweep_start_freq.setValidator(QDoubleValidator(0, 6000, 2))
        self.sweep_start_freq.setFixedWidth(100)

        self.sweep_stop_freq = QLineEdit("5700")
        self.sweep_stop_freq.setValidator(QDoubleValidator(0, 6000, 2))
        self.sweep_stop_freq.setFixedWidth(100)

        self.sweep_step_freq = QLineEdit("10")
        self.sweep_step_freq.setValidator(QDoubleValidator(0.01, 1000, 2))
        self.sweep_step_freq.setFixedWidth(100)

        self.sweep_delay = QLineEdit("100")
        self.sweep_delay.setValidator(QDoubleValidator(1, 10000, 0))
        self.sweep_delay.setFixedWidth(100)

        self.sweep_enable_btn = QPushButton("Start Sweep Mode")
        self.sweep_enable_btn.setCheckable(True)
        self.sweep_enable_btn.clicked.connect(self.toggle_sweep_mode)
        self.sweep_enable_btn.setFixedWidth(200)

        layout.addWidget(QLabel("Sweep Start Freq (MHz):"))
        layout.addWidget(self.sweep_start_freq)
        layout.addWidget(QLabel("Sweep Stop Freq (MHz):"))
        layout.addWidget(self.sweep_stop_freq)
        layout.addWidget(QLabel("Sweep Step (MHz):"))
        layout.addWidget(self.sweep_step_freq)
        layout.addWidget(QLabel("Delay per step (ms):"))
        layout.addWidget(self.sweep_delay)
        layout.addWidget(self.sweep_enable_btn)

        layout.addStretch()
        self.tx_tab.setLayout(layout)

    # ========================================================================
    # RX Methods
    # ========================================================================

    def apply_rx_settings(self):
        """Apply RX settings from UI"""
        try:
            # Stop scanning completely
            was_scanning = self.scan_timer.isActive()
            if was_scanning:
                self.scan_timer.stop()

            # Stop and wait for scan worker to finish
            if self.scan_worker is not None:
                self.scan_worker.stop()

            # Wait for thread to finish with timeout
            try:
                if self.scan_thread is not None and self.scan_thread.isRunning():
                    self.scan_thread.quit()
                    if not self.scan_thread.wait(5000):  # Wait up to 5 seconds
                        print("Warning: Scan thread did not finish in time")
                        # Force terminate if necessary
                        self.scan_thread.terminate()
                        self.scan_thread.wait(1000)
            except RuntimeError:
                pass

            # Reset thread references
            self.scan_thread = None
            self.scan_worker = None

            # Reset scanning state
            with self.scan_lock:
                self.is_scanning = False

            # Wait for hardware to settle
            time.sleep(0.3)

            # Update configuration
            self.config.start_freq = float(self.start_freq_input.text()) * 1e6
            self.config.end_freq = float(self.end_freq_input.text()) * 1e6
            self.config.step = float(self.step_input.text()) * 1e6
            self.config.span = float(self.span_input.text()) * 1e6
            self.config.sample_rate = float(self.sample_rate_input.text()) * 1e6
            self.config.num_samples = int(float(self.num_samples_input.text()))
            self.config.gain = int(float(self.gain_input.text()))

            # Validate configuration
            is_valid, error_msg = self.config.validate()
            if not is_valid:
                QMessageBox.warning(self, "Invalid Settings", error_msg)
                return

            # Reinitialize arrays
            num_points = len(np.arange(
                self.config.start_freq,
                self.config.end_freq,
                self.config.step
            )) * self.config.num_samples

            # Clean up old arrays
            if hasattr(self, 'full_spectrum'):
                del self.full_spectrum
                del self.full_freqs
                del self.max_hold_spectrum
                del self.waterfall_data

            # Create new arrays
            self.full_spectrum = np.zeros(num_points)
            self.full_freqs = np.zeros(num_points)
            self.max_hold_spectrum = np.full(num_points, -120.0)
            self.waterfall_data = np.zeros((self.config.waterfall_steps, num_points))
            self.waterfall_ptr = 0

            # Reconfigure SDR
            current_channel = self.rx_selector.currentIndex()
            self.sdr.configure_rx_channel(current_channel, self.config)

            # Update plot ranges
            self.graphWidget.setXRange(
                self.config.start_freq / 1e6,
                self.config.end_freq / 1e6
            )
            self.waterfallWidget.setYRange(0, self.config.waterfall_steps, padding=0)

            # Resume scanning if it was active
            if was_scanning:
                time.sleep(0.2)
                self.scan_timer.start(150)  # Slightly slower to prevent overload

            QMessageBox.information(self, "Success", "RX settings applied successfully")
            print("RX parameters updated successfully")

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply settings: {str(e)}")
            import traceback
            traceback.print_exc()

    def switch_rx_channel(self, index: int):
        """Switch RX channel"""
        try:
            # Stop scanning if active
            was_scanning = self.scan_timer.isActive()
            if was_scanning:
                self.scan_timer.stop()

            self.sdr.configure_rx_channel(index, self.config)
            print(f"Switched to RX{index + 1}")

            # Resume scanning if it was active
            if was_scanning:
                self.scan_timer.start(100)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch RX channel: {str(e)}")

    def toggle_scan(self):
        """Toggle spectrum scanning"""
        if self.scan_btn.isChecked():
            self.scan_btn.setText("Stop Scan")
            self.scan_timer.start(150)  # 150ms interval to prevent overload
        else:
            self.scan_btn.setText("Start Scan")
            self.scan_timer.stop()
            if self.scan_worker is not None:
                self.scan_worker.stop()
            with self.scan_lock:
                self.is_scanning = False

    def start_scan(self):
        """Start a spectrum scan"""
        # Check if already scanning
        with self.scan_lock:
            if self.is_scanning:
                return
            self.is_scanning = True

        # Don't start new scan if thread is still running
        try:
            if self.scan_thread is not None and self.scan_thread.isRunning():
                with self.scan_lock:
                    self.is_scanning = False
                return
        except RuntimeError:
            # Thread was deleted, reset reference
            self.scan_thread = None
            self.scan_worker = None

        # Create worker and thread
        self.scan_worker = ScanWorker(self.sdr, self.config)
        self.scan_thread = QThread()
        self.scan_worker.moveToThread(self.scan_thread)

        # Connect signals
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.finished.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.finished.connect(self.on_scan_thread_finished)
        self.scan_worker.result.connect(self.update_spectrum_display)
        self.scan_worker.error.connect(self.handle_scan_error)

        # Start thread
        self.scan_thread.start()

    def on_scan_thread_finished(self):
        """Clean up scan thread references after it finishes"""
        self.scan_thread = None
        self.scan_worker = None
        with self.scan_lock:
            self.is_scanning = False

    def update_spectrum_display(self, freqs: np.ndarray, spectrum: np.ndarray):
        """Update spectrum display (called from worker signal)"""
        try:
            # Store data
            self.full_freqs = freqs
            self.full_spectrum = spectrum

            # Update main spectrum plot
            self.spectrum_curve.setData(freqs / 1e6, spectrum)

            # Update max hold
            if self.max_hold_enabled:
                self.max_hold_spectrum = np.maximum(self.max_hold_spectrum, spectrum)
                self.max_hold_curve.setData(freqs / 1e6, self.max_hold_spectrum)

            # Update peak marker
            max_idx = np.argmax(spectrum)
            max_freq = freqs[max_idx] / 1e6
            max_power = spectrum[max_idx]
            self.max_marker.setData([max_freq], [max_power])
            self.max_text.setText(f"Peak: {max_power:.1f} dBm @ {max_freq:.2f} MHz")
            self.max_text.setPos(max_freq, max_power)

            # Update waterfall
            self.update_waterfall(spectrum)

        except Exception as e:
            print(f"Error updating display: {e}")
            import traceback
            traceback.print_exc()

    def update_waterfall(self, spectrum: np.ndarray):
        """Update waterfall display"""
        try:
            # Add new data to buffer
            self.waterfall_data[self.waterfall_ptr, :] = spectrum
            self.waterfall_ptr = (self.waterfall_ptr + 1) % self.waterfall_data.shape[0]

            # Display data (roll to show newest at bottom)
            img_data = np.roll(self.waterfall_data, -self.waterfall_ptr, axis=0)
            img_data = np.flipud(img_data)

            # Adjust these levels for sensitivity:
            # levels=(min_dBm, max_dBm)
            # - Lower min_dBm = see weaker signals (but more noise)
            # - Narrower range = more sensitive to small changes
            # Current: -85 to 0 dBm
            # For more sensitivity try: (-100, -20) or (-90, -10)
            self.waterfallImg.setImage(img_data.T, autoLevels=False, levels=(-85, 0))

            # Set axes
            if len(self.full_freqs) > 0:
                self.waterfallImg.setRect(QtCore.QRectF(
                    self.full_freqs[0] / 1e6,
                    0,
                    (self.full_freqs[-1] - self.full_freqs[0]) / 1e6,
                    self.waterfall_data.shape[0]
                ))

            # Sync X axis with spectrum plot
            self.waterfallWidget.setYRange(0, self.waterfall_data.shape[0], padding=0)
            self.waterfallWidget.setXRange(*self.graphWidget.viewRange()[0])

        except Exception as e:
            print(f"Error updating waterfall: {e}")

    def toggle_max_hold(self):
        """Toggle max hold feature"""
        self.max_hold_enabled = not self.max_hold_enabled
        if self.max_hold_enabled:
            self.toggle_maxhold_btn.setText("Max Hold: ON")
        else:
            self.toggle_maxhold_btn.setText("Max Hold: OFF")
            self.max_hold_spectrum[:] = -120.0
            self.max_hold_curve.setData([], [])

    def handle_scan_error(self, error_msg: str):
        """Handle scan errors"""
        QMessageBox.warning(self, "Scan Error", error_msg)
        self.scan_btn.setChecked(False)
        self.scan_btn.setText("Start Scan")
        self.scan_timer.stop()

    # ========================================================================
    # TX Methods
    # ========================================================================

    def switch_tx_channel(self, index: int):
        """Switch TX channel"""
        try:
            # Stop transmission if active
            was_transmitting = self.start_tx_btn.isChecked()
            if was_transmitting:
                self.start_tx_btn.setChecked(False)
                self.toggle_transmission()

            self.sdr.configure_tx_channel(index, self.config)
            print(f"Switched to TX{index + 1}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch TX channel: {str(e)}")

    def toggle_transmission(self):
        """Toggle single tone transmission"""
        if self.start_tx_btn.isChecked():
            try:
                # Stop sweep if running
                if self.sweep_enable_btn.isChecked():
                    self.sweep_enable_btn.setChecked(False)
                    self.toggle_sweep_mode()
                    time.sleep(0.3)  # Wait for sweep to stop

                freq = float(self.tx_freq_input.text()) * 1e6
                power = float(self.tx_power_input.text())

                # Validate inputs
                if freq < 1e6 or freq > 6000e6:
                    raise ValueError("Frequency must be between 1 and 6000 MHz")
                if power < 0 or power > 30:
                    raise ValueError("Power must be between 0 and 30 dB")

                # Reconfigure TX sync to ensure it's initialized
                with self.sdr.lock:
                    print("Configuring TX sync for single tone...")
                    self.sdr.sdr.sync_config(
                        layout=_bladerf.ChannelLayout.TX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=self.config.SYNC_NUM_BUFFERS,
                        buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                        num_transfers=self.config.SYNC_NUM_TRANSFERS,
                        stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                    )

                    if self.sdr.tx is not None:
                        self.sdr.tx.enable = True
                    time.sleep(0.1)

                # Generate test signal (1 MHz tone)
                t = np.arange(self.config.num_samples) / self.config.sample_rate
                tx_signal = 0.7 * np.exp(2j * np.pi * 1e6 * t)

                # Transmit
                self.sdr.transmit_signal(freq, power, tx_signal)

                # Update labels
                self.tx_freq_label.setText(f"TX Frequency: {freq / 1e6:.2f} MHz")
                self.tx_power_label.setText(f"TX Power: {int(round(power))} dB")
                self.start_tx_btn.setText("Stop Transmission")

                print(f"Started transmission at {freq / 1e6:.2f} MHz with {int(round(power))} dB power")

            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", str(e))
                self.start_tx_btn.setChecked(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start transmission: {str(e)}")
                self.start_tx_btn.setChecked(False)
                import traceback
                traceback.print_exc()
        else:
            try:
                with self.sdr.lock:
                    if self.sdr.tx is not None:
                        self.sdr.tx.enable = False
                self.start_tx_btn.setText("Start Transmission")
                print("Transmission stopped")
            except Exception as e:
                print(f"Error stopping transmission: {e}")

    def toggle_sweep_mode(self):
        """Toggle sweep mode"""
        if self.sweep_enable_btn.isChecked():
            try:
                # Stop single tone transmission if running
                if self.start_tx_btn.isChecked():
                    self.start_tx_btn.setChecked(False)
                    self.toggle_transmission()
                    time.sleep(0.3)  # Wait for transmission to stop

                start_freq = float(self.sweep_start_freq.text()) * 1e6
                stop_freq = float(self.sweep_stop_freq.text()) * 1e6
                step_freq = float(self.sweep_step_freq.text()) * 1e6
                delay_ms = int(float(self.sweep_delay.text()))
                power = float(self.tx_power_input.text())

                # Validate inputs
                if start_freq >= stop_freq:
                    raise ValueError("Start frequency must be less than stop frequency")
                if step_freq <= 0 or step_freq > (stop_freq - start_freq):
                    raise ValueError("Invalid step frequency")
                if delay_ms < 1:
                    raise ValueError("Delay must be at least 1 ms")

                # Create worker and thread
                self.sweep_worker = SweepWorker(
                    self.sdr, self.config,
                    start_freq, stop_freq, step_freq,
                    delay_ms, power
                )
                self.sweep_thread = QThread()
                self.sweep_worker.moveToThread(self.sweep_thread)

                # Connect signals
                self.sweep_thread.started.connect(self.sweep_worker.run)
                self.sweep_worker.finished.connect(self.sweep_thread.quit)
                self.sweep_worker.finished.connect(self.sweep_worker.deleteLater)
                self.sweep_thread.finished.connect(self.sweep_thread.deleteLater)
                self.sweep_thread.finished.connect(self.on_sweep_thread_finished)
                self.sweep_worker.status_update.connect(self.update_sweep_status)
                self.sweep_worker.error.connect(self.handle_sweep_error)

                # Start thread
                self.sweep_thread.start()
                self.sweep_enable_btn.setText("Stop Sweep Mode")

                print(f"Started sweep from {start_freq/1e6:.2f} to {stop_freq/1e6:.2f} MHz")

            except ValueError as e:
                QMessageBox.warning(self, "Invalid Input", str(e))
                self.sweep_enable_btn.setChecked(False)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to start sweep: {str(e)}")
                self.sweep_enable_btn.setChecked(False)
        else:
            if self.sweep_worker is not None:
                self.sweep_worker.stop()
            self.sweep_enable_btn.setText("Start Sweep Mode")

    def on_sweep_thread_finished(self):
        """Clean up sweep thread references after it finishes"""
        self.sweep_thread = None
        self.sweep_worker = None

        # Disable TX after sweep
        try:
            with self.sdr.lock:
                if self.sdr.tx is not None:
                    self.sdr.tx.enable = False
            print("TX disabled after sweep")
        except Exception as e:
            print(f"Error disabling TX after sweep: {e}")

    def update_sweep_status(self, message: str, freq: float):
        """Update sweep status (called from worker signal)"""
        print(message)
        self.tx_freq_label.setText(f"TX Frequency: {freq / 1e6:.2f} MHz")

    def handle_sweep_error(self, error_msg: str):
        """Handle sweep errors"""
        QMessageBox.warning(self, "Sweep Error", error_msg)
        self.sweep_enable_btn.setChecked(False)
        self.sweep_enable_btn.setText("Start Sweep Mode")

    # ========================================================================
    # Cleanup
    # ========================================================================

    def closeEvent(self, event):
        """Handle application close"""
        # Stop all operations
        if self.scan_timer.isActive():
            self.scan_timer.stop()

        if self.scan_worker is not None:
            self.scan_worker.stop()

        if self.sweep_worker is not None:
            self.sweep_worker.stop()

        # Wait for threads to finish
        try:
            if self.scan_thread is not None and self.scan_thread.isRunning():
                self.scan_thread.quit()
                self.scan_thread.wait(2000)
        except RuntimeError:
            # Thread already deleted
            pass

        try:
            if self.sweep_thread is not None and self.sweep_thread.isRunning():
                self.sweep_thread.quit()
                self.sweep_thread.wait(2000)
        except RuntimeError:
            # Thread already deleted
            pass

        # Close SDR
        self.sdr.close()

        event.accept()


# ============================================================================
# Logging System
# ============================================================================

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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point"""
    # Set locale
    QLocale.setDefault(QLocale("C"))

    # Determine base path
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    # Setup logging
    log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
    log_path = os.path.join(base_path, log_filename)

    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print(f"Log started: {log_path}")
    print(f"BladeRF Spectrum Analyzer v2.0 (Refactored)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create application
    app = QApplication(sys.argv)

    # Set icon
    if getattr(sys, 'frozen', False):
        icon_path = os.path.join(sys._MEIPASS, "bladerf2_0.ico")
    else:
        icon_path = os.path.join(base_path, "bladerf2_0.ico")

    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Create and show main window
    try:
        window = SpectrumAnalyzer()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        QMessageBox.critical(None, "Fatal Error",
                           f"Application failed to start:\n{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()