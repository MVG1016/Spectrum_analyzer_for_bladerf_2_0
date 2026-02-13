"""
BladeRF 2.0 Spectrum Analyzer - Optimized Architecture
With channel selection and logging
"""

import numpy as np
import time
import sys
import os
from datetime import datetime
from threading import Lock
from dataclasses import dataclass
from typing import Optional, Dict

from bladerf import _bladerf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                            QPushButton, QHBoxLayout, QComboBox, QFormLayout,
                            QLineEdit, QLabel, QMessageBox, QSpinBox, QFrame)
from PyQt5.QtCore import QTimer, QLocale, QThread, QRectF
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
import pyqtgraph as pg


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
# Configuration
# ============================================================================

@dataclass
class SDRConfig:
    """SDR configuration parameters"""
    start_freq: float = 70e6      # 70 MHz
    stop_freq: float = 6000e6     # 6000 MHz
    step: float = 0.53e6          # 0.53 MHz (equal to sample rate)

    sample_rate: float = step     # 0.53 MHz
    num_samples: int = 4096
    gain: int = 30

    # Waterfall
    waterfall_lines: int = 30

    # BladeRF streaming
    SYNC_NUM_BUFFERS: int = 16
    SYNC_NUM_TRANSFERS: int = 8
    SYNC_STREAM_TIMEOUT: int = 3500
    BUFFER_SIZE_MULTIPLIER: int = 4


# Calibration table
CALIBRATION_TABLE = {
    400e6: -77,
    800e6: -77,
    1500e6: -77,
    3000e6: -77,
    5000e6: -77
}


def get_calibration(freq_hz: float) -> float:
    """Get calibration offset for frequency"""
    freqs = np.array(list(CALIBRATION_TABLE.keys()))
    gains = np.array(list(CALIBRATION_TABLE.values()))
    return np.interp(freq_hz, freqs, gains)


# ============================================================================
# TX Thread for Continuous Transmission
# ============================================================================

class TXThread(QThread):
    """Continuous TX transmission thread"""

    def __init__(self, sdr, tx_buffer, parent=None):
        super().__init__(parent)
        self.sdr = sdr
        self.tx_buffer = tx_buffer
        self.running = True

    def run(self):
        """Continuous transmission loop"""
        while self.running:
            try:
                # Transmit buffer continuously
                self.sdr.sync_tx(self.tx_buffer, len(self.tx_buffer) // 4)
                time.sleep(0.001)  # Small delay to prevent CPU overload
            except Exception as e:
                print(f"TX send error: {e}")
                time.sleep(0.01)

    def stop(self):
        """Stop transmission"""
        self.running = False
        self.wait()


# ============================================================================
# Main Spectrum Analyzer
# ============================================================================

class SpectrumAnalyzer(QMainWindow):
    """Main spectrum analyzer window"""

    def __init__(self):
        super().__init__()

        print("Initializing main window...")

        self.setWindowTitle("BladeRF 2.0 Wideband Spectrum Analyzer")
        self.setGeometry(100, 100, 1400, 900)

        # Configuration
        self.config = SDRConfig()
        print("Configuration created")

        # BladeRF device
        self.sdr = None
        self.rx_channel = None
        self.tx_channel = None
        self.sdr_lock = Lock()

        # Current channel indices
        self.current_rx_channel_index = 0
        self.current_tx_channel_index = 0

        # RX state
        self.live_scanning = False
        self.center_freq = 1000e6
        self.buffer = None

        # Composite scanning
        self.wb_centers = None       # Center frequencies for sweep
        self.wb_index = 0
        self.composite_spectrum = None
        self.common_freq = None      # Common frequency axis

        # Max Hold
        self.maxhold_enabled = False
        self.maxhold_data_arr = None

        # Waterfall
        self.waterfall_data = None
        self.waterfall_index = 0

        # TX state
        self.tx_enabled = False
        self.tx_thread = None
        self.tx_buffer = None

        # Sweep mode
        self.sweep_enabled = False
        self.sweep_timer = QTimer()
        self.sweep_timer.timeout.connect(self.next_sweep_step)
        self.sweep_freqs = []
        self.current_sweep_freq = 0

        # Current spectrum for cursor
        self.current_x = None
        self.current_y = None

        print("State variables initialized")

        # Initialize UI
        try:
            print("Creating UI...")
            self.init_ui()
            print("UI created successfully")
        except Exception as e:
            print(f"ERROR creating UI: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Initialize BladeRF AFTER window is shown (delayed)
        print("Scheduling BladeRF initialization...")
        QTimer.singleShot(100, self.init_bladerf_delayed)

    def init_bladerf_delayed(self):
        """Initialize BladeRF after window is shown"""
        try:
            print("Starting delayed BladeRF initialization...")
            self.init_bladerf()
            print("BladeRF initialized successfully")
        except Exception as e:
            print(f"ERROR initializing BladeRF: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "BladeRF Error",
                f"Failed to initialize BladeRF:\n{str(e)}\n\nPlease check device connection."
            )

    def init_ui(self):
        """Initialize user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ===== LEFT: Graphs =====
        graph_container = QWidget()
        graph_layout = QVBoxLayout(graph_container)

        # Spectrum plot
        self.graph_plot = pg.PlotWidget(title="Spectrum in dBm")
        self.graph_plot.setLabel('left', 'Power', units='dBm')
        self.graph_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.graph_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.graph_plot.setYRange(-140, -30)
        self.graph_plot.showGrid(x=True, y=True)

        # Main spectrum curve (yellow)
        self.curve = self.graph_plot.plot(pen='y')

        # Max hold curve (red)
        self.maxhold_curve = self.graph_plot.plot(pen='r')

        # Peak marker
        self.max_marker = self.graph_plot.plot(
            symbol='o', symbolBrush='r', symbolSize=10
        )
        self.max_text = pg.TextItem(color='w', anchor=(0, 1))
        self.graph_plot.addItem(self.max_text)

        # Cursor crosshair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('c'))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('c'))
        self.graph_plot.addItem(self.vLine, ignoreBounds=True)
        self.graph_plot.addItem(self.hLine, ignoreBounds=True)
        self.cursorLabel = pg.TextItem("", anchor=(1, 1),
                                       fill=pg.mkBrush(0, 0, 0, 150))
        self.graph_plot.addItem(self.cursorLabel)

        # Connect mouse movement
        self.proxy = pg.SignalProxy(
            self.graph_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.on_mouse_moved
        )

        graph_layout.addWidget(self.graph_plot)

        # Waterfall plot
        self.waterfall_plot = pg.PlotWidget(title="Waterfall")
        self.waterfall_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.waterfall_plot.setLabel('left', 'Scan')
        self.waterfall_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.waterfall_img = pg.ImageItem()
        self.waterfall_img.setOpts(invertY=False, axisOrder='row-major')

        # Colormap
        lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        self.waterfall_img.setLookupTable(lut)
        self.waterfall_plot.addItem(self.waterfall_img)
        self.waterfall_plot.getViewBox().invertY(True)

        graph_layout.addWidget(self.waterfall_plot)
        main_layout.addWidget(graph_container, stretch=3)

        # ===== RIGHT: Control Panel =====
        control_panel = QWidget()
        control_layout = QFormLayout(control_panel)

        # RX Settings
        control_layout.addRow(QLabel("<b>Receiver Settings</b>"))

        # RX Channel selector
        self.rx_channel_combo = QComboBox()
        self.rx_channel_combo.addItems(["RX1", "RX2"])
        self.rx_channel_combo.currentIndexChanged.connect(self.on_rx_channel_changed)
        control_layout.addRow("RX Channel:", self.rx_channel_combo)

        self.start_freq_edit = QLineEdit("2300")
        control_layout.addRow("Start (MHz):", self.start_freq_edit)

        self.stop_freq_edit = QLineEdit("2500")
        control_layout.addRow("Stop (MHz):", self.stop_freq_edit)

        self.step_edit = QLineEdit("50")
        control_layout.addRow("Step (MHz):", self.step_edit)

        self.samples_combo = QComboBox()
        for v in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
            self.samples_combo.addItem(str(v))
        self.samples_combo.setCurrentText("4096")
        control_layout.addRow("FFT size:", self.samples_combo)

        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 60)
        self.gain_spin.setValue(30)
        control_layout.addRow("Gain:", self.gain_spin)

        self.waterfall_lines_spin = QSpinBox()
        self.waterfall_lines_spin.setRange(10, 2000)
        self.waterfall_lines_spin.setValue(30)
        control_layout.addRow("Waterfall size:", self.waterfall_lines_spin)

        self.scan_button = QPushButton("Start Scanning")
        self.scan_button.clicked.connect(self.toggle_live_scanning)
        control_layout.addRow(self.scan_button)

        self.maxhold_button = QPushButton("Turn Max Hold On")
        self.maxhold_button.clicked.connect(self.toggle_maxhold)
        control_layout.addRow(self.maxhold_button)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(separator1)

        # TX Settings
        control_layout.addRow(QLabel("<b>Transmitter Settings</b>"))

        # TX Channel selector
        self.tx_channel_combo = QComboBox()
        self.tx_channel_combo.addItems(["TX1", "TX2"])
        self.tx_channel_combo.currentIndexChanged.connect(self.on_tx_channel_changed)
        control_layout.addRow("TX Channel:", self.tx_channel_combo)

        self.tx_freq_edit = QLineEdit("2400")
        control_layout.addRow("TX Frequency (MHz):", self.tx_freq_edit)

        self.tx_gain_spin = QSpinBox()
        self.tx_gain_spin.setRange(0, 60)
        self.tx_gain_spin.setValue(10)
        control_layout.addRow("TX Gain:", self.tx_gain_spin)

        self.tx_start_button = QPushButton("Start Transmission")
        self.tx_start_button.clicked.connect(self.start_transmission)
        control_layout.addRow(self.tx_start_button)

        self.tx_status_label = QLabel("")
        control_layout.addRow("Status:", self.tx_status_label)

        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(separator2)

        # Sweep Settings
        control_layout.addRow(QLabel("<b>Sweep Mode</b>"))

        self.sweep_start_edit = QLineEdit("2300")
        control_layout.addRow("Sweep Start (MHz):", self.sweep_start_edit)

        self.sweep_stop_edit = QLineEdit("2500")
        control_layout.addRow("Sweep Stop (MHz):", self.sweep_stop_edit)

        self.sweep_step_edit = QLineEdit("10")
        control_layout.addRow("Sweep Step (MHz):", self.sweep_step_edit)

        self.sweep_dwell_edit = QLineEdit("100")
        control_layout.addRow("Dwell Time (ms):", self.sweep_dwell_edit)

        self.sweep_gain_spin = QSpinBox()
        self.sweep_gain_spin.setRange(0, 60)
        self.sweep_gain_spin.setValue(30)
        control_layout.addRow("Sweep Gain:", self.sweep_gain_spin)

        self.sweep_start_button = QPushButton("Start Sweep")
        self.sweep_start_button.clicked.connect(self.toggle_sweep_transmission)
        control_layout.addRow(self.sweep_start_button)

        self.sweep_status_label = QLabel("")
        control_layout.addRow("Sweep Status:", self.sweep_status_label)

        main_layout.addWidget(control_panel, stretch=1)

    def init_bladerf(self):
        """Initialize BladeRF device"""
        try:
            print("Opening BladeRF device...")
            with self.sdr_lock:
                self.sdr = _bladerf.BladeRF()
                print(f"BladeRF device opened: {self.sdr.device_speed}")

                # Initialize RX channel
                print("Initializing RX channel...")
                self.init_rx_channel(self.current_rx_channel_index)
                print("RX channel initialized")

                # Initialize buffer
                print("Allocating buffer...")
                self.buffer = np.zeros(self.config.num_samples, dtype=np.complex64)
                print(f"Buffer allocated: {self.config.num_samples} samples")

                print("BladeRF initialization complete")

        except Exception as e:
            print(f"CRITICAL ERROR in init_bladerf: {e}")
            import traceback
            traceback.print_exc()
            raise

    def init_rx_channel(self, channel_index: int):
        """Initialize RX channel"""
        print(f"Initializing RX channel {channel_index}...")

        # Disable old channel if exists
        if self.rx_channel is not None:
            try:
                print("Disabling old RX channel...")
                self.rx_channel.enable = False
                print("Old RX channel disabled")
            except Exception as e:
                print(f"Warning disabling old RX channel: {e}")
            time.sleep(0.2)

        # Create new channel - try different approach
        print(f"Creating RX channel object for index {channel_index}...")
        try:
            # Method 1: Direct creation
            rx_ch = _bladerf.CHANNEL_RX(channel_index)
            print(f"CHANNEL_RX enum created: {rx_ch}")

            self.rx_channel = self.sdr.Channel(rx_ch)
            print("RX channel object created successfully")
        except Exception as e:
            print(f"ERROR creating channel with Method 1: {e}")
            # Method 2: Try with explicit int
            try:
                print("Trying alternate method...")
                if channel_index == 0:
                    self.rx_channel = self.sdr.Channel(_bladerf.CHANNEL_RX1)
                else:
                    self.rx_channel = self.sdr.Channel(_bladerf.CHANNEL_RX2)
                print("RX channel created with alternate method")
            except Exception as e2:
                print(f"ERROR with alternate method: {e2}")
                raise

        # Configure channel
        print("Setting RX sample rate...")
        self.rx_channel.sample_rate = int(self.config.sample_rate)
        print(f"  Sample rate set: {self.config.sample_rate}")

        print("Setting RX bandwidth...")
        self.rx_channel.bandwidth = int(self.config.sample_rate)
        print(f"  Bandwidth set: {self.config.sample_rate}")

        print("Setting RX gain mode...")
        self.rx_channel.gain_mode = _bladerf.GainMode.Manual
        print("  Gain mode set: Manual")

        print("Setting RX gain...")
        self.rx_channel.gain = self.config.gain
        print(f"  Gain set: {self.config.gain}")

        print("Setting RX frequency...")
        self.rx_channel.frequency = int(self.center_freq)
        print(f"  Frequency set: {self.center_freq}")

        # Configure RX streaming
        print("Configuring RX streaming...")
        self.sdr.sync_config(
            layout=_bladerf.ChannelLayout.RX_X1,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=self.config.SYNC_NUM_BUFFERS,
            buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
            num_transfers=self.config.SYNC_NUM_TRANSFERS,
            stream_timeout=self.config.SYNC_STREAM_TIMEOUT
        )
        print("RX sync configured")

        print("Enabling RX channel...")
        self.rx_channel.enable = True
        print("RX channel enabled")

        time.sleep(0.1)

        self.current_rx_channel_index = channel_index
        print(f"RX Channel {channel_index + 1} initialized successfully")

    def init_tx_channel(self, channel_index: int):
        """Initialize TX channel"""
        try:
            with self.sdr_lock:
                # Disable old channel if exists
                if self.tx_channel is not None:
                    try:
                        self.tx_channel.enable = False
                    except Exception as e:
                        print(f"Warning disabling old TX channel: {e}")
                    time.sleep(0.2)

                # Create new channel
                self.tx_channel = self.sdr.Channel(_bladerf.CHANNEL_TX(channel_index))
                self.tx_channel.sample_rate = int(self.config.sample_rate)
                self.tx_channel.bandwidth = int(self.config.sample_rate)
                self.tx_channel.gain_mode = _bladerf.GainMode.Manual
                self.tx_channel.gain = self.tx_gain_spin.value()

                # Configure TX streaming
                self.sdr.sync_config(
                    layout=_bladerf.ChannelLayout.TX_X1,
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=self.config.SYNC_NUM_BUFFERS,
                    buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                    num_transfers=self.config.SYNC_NUM_TRANSFERS,
                    stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                )

                self.tx_channel.enable = False
                time.sleep(0.05)

                self.current_tx_channel_index = channel_index
                print(f"TX Channel {channel_index + 1} initialized")

        except Exception as e:
            print(f"Error initializing TX channel: {e}")
            import traceback
            traceback.print_exc()
            raise

    def on_rx_channel_changed(self, index: int):
        """Handle RX channel change"""
        if self.sdr is None:
            return

        # Stop scanning if active
        was_scanning = self.live_scanning
        if was_scanning:
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")

        try:
            self.init_rx_channel(index)
            print(f"Switched to RX{index + 1}")

            # Resume scanning if it was active
            if was_scanning:
                QTimer.singleShot(300, lambda: self.scan_button.click())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch RX channel: {e}")

    def on_tx_channel_changed(self, index: int):
        """Handle TX channel change"""
        if self.sdr is None:
            return

        # Stop transmission/sweep if active
        was_transmitting = self.tx_enabled
        was_sweeping = self.sweep_enabled

        if was_transmitting:
            self.start_transmission()  # Toggle off
        if was_sweeping:
            self.toggle_sweep_transmission()  # Toggle off

        try:
            self.init_tx_channel(index)
            print(f"Switched to TX{index + 1}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch TX channel: {e}")

    def acquire_one_spectrum(self) -> tuple:
        """
        Acquire one spectrum at current center frequency
        Returns: (freq_axis, power_dbm)
        """
        with self.sdr_lock:
            try:
                # Read samples
                buf = bytearray(self.config.num_samples * 4)
                self.sdr.sync_rx(buf, self.config.num_samples)

                # Convert to complex
                samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                samples = samples.view(np.complex64)
                samples /= 2048.0

                # FFT
                spectrum = np.fft.fftshift(np.fft.fft(samples))
                power = np.abs(spectrum) ** 2 / self.config.num_samples

                # Calibration
                cal_offset = get_calibration(self.center_freq)
                power_dbm = 10 * np.log10(power / 1e-3 + 1e-12) + cal_offset

                # Frequency axis
                f_start = (self.center_freq - self.config.sample_rate / 2) / 1e6
                f_end = (self.center_freq + self.config.sample_rate / 2) / 1e6
                freq_axis = np.linspace(f_start, f_end, self.config.num_samples,
                                       endpoint=True)

                return freq_axis, power_dbm

            except Exception as e:
                print(f"Error acquiring spectrum: {e}")
                # Return dummy data
                freq_axis = np.linspace(0, 1, self.config.num_samples)
                power_dbm = np.full(self.config.num_samples, -140.0)
                return freq_axis, power_dbm

    def toggle_live_scanning(self):
        """Toggle live scanning mode"""
        if self.live_scanning:
            # Stop scanning
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
        else:
            # Start scanning
            try:
                # Update parameters
                self.config.num_samples = int(self.samples_combo.currentText())
                self.config.waterfall_lines = self.waterfall_lines_spin.value()
                self.config.gain = self.gain_spin.value()

                # Reconfigure RX
                with self.sdr_lock:
                    self.rx_channel.enable = False
                    time.sleep(0.1)

                    self.rx_channel.gain = self.config.gain

                    self.sdr.sync_config(
                        layout=_bladerf.ChannelLayout.RX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=self.config.SYNC_NUM_BUFFERS,
                        buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                        num_transfers=self.config.SYNC_NUM_TRANSFERS,
                        stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                    )

                    self.rx_channel.enable = True
                    time.sleep(0.05)

                # Initialize waterfall
                self.waterfall_data = np.full(
                    (self.config.waterfall_lines, self.config.num_samples),
                    -140.0
                )
                self.waterfall_index = 0
                self.waterfall_img.setImage(
                    self.waterfall_data,
                    autoLevels=False,
                    levels=(-140, -30)
                )

                # Start scanning
                self.live_scanning = True
                self.scan_button.setText("Pause")
                self.init_live_scan_parameters()
                self.composite_scan_cycle()

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start scanning: {e}")
                import traceback
                traceback.print_exc()
                self.live_scanning = False

    def init_live_scan_parameters(self):
        try:
            start_mhz = float(self.start_freq_edit.text())
            stop_mhz = float(self.stop_freq_edit.text())
            step_mhz = float(self.step_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid frequency values")
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
            return
        overlap = 1.1
        self.config.sample_rate = step_mhz * 1e6  *overlap

        with self.sdr_lock:
            self.rx_channel.sample_rate = int(self.config.sample_rate)
            self.rx_channel.bandwidth = int(self.config.sample_rate)

        start_hz = start_mhz * 1e6
        stop_hz = stop_mhz * 1e6
        step_hz = step_mhz * 1e6

        self.wb_centers = np.arange(
            start_hz + self.config.sample_rate / 2,
            stop_hz - self.config.sample_rate / 2 + step_hz,
            step_hz
        )

        self.wb_index = 0

        # Общая частотная ось
        common_res = 0.1  # MHz
        num_points = int(np.round((stop_mhz - start_mhz) / common_res)) + 1
        self.common_freq = np.linspace(start_mhz, stop_mhz, num_points)

        # Один sweep = один кадр
        self.composite_spectrum = np.full(self.common_freq.shape, -140.0)

        # Параметры realtime сегмента
        self.segment_accum = None
        self.segment_count = 0
        self.segment_avg_len = 20  # ВАЖНО: dwell time

        # Waterfall
        self.waterfall_img.setRect(
            QRectF(start_mhz, 0, stop_mhz - start_mhz, self.config.waterfall_lines)
        )

        self.curve.clear()
        self.waterfall_data.fill(-140)
        self.waterfall_index = 0

        print(f"Realtime scan: {start_mhz:.1f}-{stop_mhz:.1f} MHz, "
              f"step {step_mhz:.1f} MHz, {len(self.wb_centers)} points")

    def composite_scan_cycle(self):
        if not self.live_scanning:
            return

        if self.wb_index >= len(self.wb_centers):
            self.update_display()
            self.wb_index = 0
            QTimer.singleShot(0, self.composite_scan_cycle)
            return

        new_center = self.wb_centers[self.wb_index]

        with self.sdr_lock:
            self.rx_channel.frequency = int(new_center)
            self.center_freq = new_center

        # сброс сегмента
        self.segment_accum = None
        self.segment_count = 0

        QTimer.singleShot(2, self.do_composite_measurement)

    def do_composite_measurement(self):
        if not self.live_scanning:
            return

        try:
            meas_freq, meas_power = self.acquire_one_spectrum()

            if self.segment_accum is None:
                self.segment_accum = meas_power.copy()
            else:
                self.segment_accum += meas_power

            self.segment_count += 1

            # ещё наблюдаем эту частоту
            if self.segment_count < self.segment_avg_len:
                QTimer.singleShot(0, self.do_composite_measurement)
                return

            # формируем сегмент
            avg_power = self.segment_accum / self.segment_count

            seg_min = meas_freq[0]
            seg_max = meas_freq[-1]
            mask = (self.common_freq >= seg_min) & (self.common_freq <= seg_max)

            if np.any(mask):
                interp_power = np.interp(
                    self.common_freq[mask],
                    meas_freq,
                    avg_power,
                    left=-140,
                    right=-140
                )

                # Clear/Write — НОРМАЛЬНО для realtime
                self.composite_spectrum[mask] = interp_power

                if self.maxhold_enabled:
                    if self.maxhold_data_arr is None:
                        self.maxhold_data_arr = self.composite_spectrum.copy()
                    else:
                        self.maxhold_data_arr[mask] = np.maximum(
                            self.maxhold_data_arr[mask],
                            interp_power
                        )



        except Exception as e:
            print(f"Measurement error: {e}")

        self.wb_index += 1
        QTimer.singleShot(0, self.composite_scan_cycle)

    def update_display(self):
        try:
            self.curve.setData(self.common_freq, self.composite_spectrum)
            self.current_x = self.common_freq.copy()
            self.current_y = self.composite_spectrum.copy()

            if self.maxhold_enabled and self.maxhold_data_arr is not None:
                self.maxhold_curve.setData(self.common_freq, self.maxhold_data_arr)
            else:
                self.maxhold_curve.clear()

            if self.composite_spectrum.size > 0:
                idx = np.argmax(self.composite_spectrum)
                f = self.common_freq[idx]
                p = self.composite_spectrum[idx]

                self.max_marker.setData([f], [p])
                self.max_text.setText(f"{f:.2f} MHz\n{p:.1f} dBm")
                self.max_text.setPos(f, p)

            row_data = np.interp(
                np.linspace(self.common_freq[0], self.common_freq[-1],
                            self.config.num_samples),
                self.common_freq,
                self.composite_spectrum
            )

            if self.waterfall_index < self.config.waterfall_lines:
                self.waterfall_data[self.waterfall_index] = row_data
                self.waterfall_index += 1
            else:
                self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
                self.waterfall_data[-1] = row_data

            self.waterfall_img.setImage(
                self.waterfall_data,
                autoLevels=False,
                levels=(-140, -30)
            )

        except Exception as e:
            print(f"Display error: {e}")

    def toggle_maxhold(self):
        """Toggle max hold feature"""
        self.maxhold_enabled = not self.maxhold_enabled

        if self.maxhold_enabled:
            self.maxhold_button.setText("Turn Max Hold Off")
            if self.common_freq is not None:
                self.maxhold_data_arr = self.composite_spectrum.copy()
        else:
            self.maxhold_button.setText("Turn Max Hold On")
            self.maxhold_curve.clear()

    def start_transmission(self):
        """Start/stop single tone transmission"""
        if not self.tx_enabled:
            try:
                # Get parameters
                tx_freq = float(self.tx_freq_edit.text()) * 1e6
                tx_gain = self.tx_gain_spin.value()

                # Initialize TX channel if needed
                if self.tx_channel is None:
                    self.init_tx_channel(self.current_tx_channel_index)

                # IMPORTANT: Reconfigure for TX
                print("Configuring for TX transmission...")
                with self.sdr_lock:
                    # Disable channels temporarily
                    # if self.rx_channel is not None:
                    #     self.rx_channel.enable = False
                    if self.tx_channel is not None:
                        self.tx_channel.enable = False

                    time.sleep(0.1)

                    # Configure TX streaming (separate from RX)
                    print("Configuring TX streaming...")
                    self.sdr.sync_config(
                        layout=_bladerf.ChannelLayout.TX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=self.config.SYNC_NUM_BUFFERS,
                        buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                        num_transfers=self.config.SYNC_NUM_TRANSFERS,
                        stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                    )
                    print("TX sync configured")

                    # Configure and enable TX
                    self.tx_channel.frequency = int(tx_freq)
                    self.tx_channel.gain = tx_gain
                    self.tx_channel.enable = True
                    time.sleep(0.05)

                # Generate tone (1 kHz)
                t = np.arange(self.config.num_samples)
                tone_freq = 1000
                signal = np.exp(1j * 2 * np.pi * tone_freq * t / self.config.sample_rate)

                # Convert to SC16
                iq = np.empty(2 * len(signal), dtype=np.int16)
                iq[0::2] = np.clip(np.real(signal) * 2047, -2048, 2047).astype(np.int16)
                iq[1::2] = np.clip(np.imag(signal) * 2047, -2048, 2047).astype(np.int16)
                self.tx_buffer = iq.tobytes()

                # Start TX thread
                self.tx_thread = TXThread(self.sdr, self.tx_buffer)
                self.tx_thread.start()

                self.tx_enabled = True
                self.tx_start_button.setText("Stop Transmission")
                self.tx_status_label.setText(
                    f"TX{self.current_tx_channel_index + 1}: {tx_freq/1e6:.2f} MHz, Gain {tx_gain}"
                )

                print(f"TX started on channel {self.current_tx_channel_index + 1} at {tx_freq/1e6:.2f} MHz")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start TX: {e}")
                import traceback
                traceback.print_exc()
                self.tx_enabled = False
        else:
            # Stop transmission
            self.tx_enabled = False

            if self.tx_thread is not None:
                self.tx_thread.stop()

            with self.sdr_lock:
                if self.tx_channel is not None:
                    self.tx_channel.enable = False

                # Reconfigure back to RX only if scanning
                if self.live_scanning and self.rx_channel is not None:
                    print("Reconfiguring back to RX only...")
                    time.sleep(0.1)

                    self.sdr.sync_config(
                        layout=_bladerf.ChannelLayout.RX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=self.config.SYNC_NUM_BUFFERS,
                        buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                        num_transfers=self.config.SYNC_NUM_TRANSFERS,
                        stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                    )

                    self.rx_channel.enable = True
                    print("RX re-enabled")

            self.tx_start_button.setText("Start Transmission")
            self.tx_status_label.setText("Transmission stopped")
            print("TX stopped")

    def toggle_sweep_transmission(self):
        """Toggle sweep transmission mode"""
        if not self.sweep_enabled:
            try:
                # Get parameters
                start_freq = float(self.sweep_start_edit.text()) * 1e6
                stop_freq = float(self.sweep_stop_edit.text()) * 1e6
                step_freq = float(self.sweep_step_edit.text()) * 1e6
                dwell_time = float(self.sweep_dwell_edit.text())

                if step_freq <= 0:
                    raise ValueError("Step must be positive")

                # Generate frequency list
                if start_freq < stop_freq:
                    self.sweep_freqs = np.arange(start_freq, stop_freq + step_freq, step_freq)
                else:
                    self.sweep_freqs = np.arange(start_freq, stop_freq - step_freq, -step_freq)

                if len(self.sweep_freqs) == 0:
                    raise ValueError("Invalid sweep parameters")

                # Initialize TX channel if needed
                if self.tx_channel is None:
                    self.init_tx_channel(self.current_tx_channel_index)

                # Configure TX gain
                tx_gain = self.sweep_gain_spin.value()

                # Configure for TX
                print("Configuring for TX sweep...")
                with self.sdr_lock:
                    # # Disable channels temporarily
                    # if self.rx_channel is not None:
                    #     self.rx_channel.enable = False
                    if self.tx_channel is not None:
                        self.tx_channel.enable = False

                    time.sleep(0.1)

                    # Configure TX streaming
                    print("Configuring TX streaming...")
                    self.sdr.sync_config(
                        layout=_bladerf.ChannelLayout.TX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=self.config.SYNC_NUM_BUFFERS,
                        buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                        num_transfers=self.config.SYNC_NUM_TRANSFERS,
                        stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                    )
                    print("TX sync configured")

                    # Configure and enable TX
                    self.tx_channel.gain = tx_gain
                    self.tx_channel.enable = True
                    time.sleep(0.05)

                # Generate tone
                t = np.arange(self.config.num_samples)
                tone_freq = 1000
                signal = np.exp(1j * 2 * np.pi * tone_freq * t / self.config.sample_rate)

                # Convert to SC16
                iq = np.empty(2 * len(signal), dtype=np.int16)
                iq[0::2] = np.clip(np.real(signal) * 2047, -2048, 2047).astype(np.int16)
                iq[1::2] = np.clip(np.imag(signal) * 2047, -2048, 2047).astype(np.int16)
                self.tx_buffer = iq.tobytes()

                # Start TX thread
                if self.tx_thread is None or not self.tx_thread.isRunning():
                    self.tx_thread = TXThread(self.sdr, self.tx_buffer)
                    self.tx_thread.start()

                # Start sweep
                self.sweep_enabled = True
                self.current_sweep_freq = 0
                self.sweep_timer.start(int(dwell_time))

                self.sweep_start_button.setText("Stop Sweep")
                self.sweep_status_label.setText(
                    f"TX{self.current_tx_channel_index + 1} Sweep: {self.sweep_freqs[0]/1e6:.2f} MHz"
                )

                # Set first frequency
                with self.sdr_lock:
                    self.tx_channel.frequency = int(self.sweep_freqs[0])

                print(f"Sweep started on TX{self.current_tx_channel_index + 1}: "
                      f"{start_freq/1e6:.1f} - {stop_freq/1e6:.1f} MHz")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start sweep: {e}")
                import traceback
                traceback.print_exc()
                self.sweep_enabled = False
        else:
            # Stop sweep
            self.sweep_enabled = False
            self.sweep_timer.stop()

            self.sweep_start_button.setText("Start Sweep")
            self.sweep_status_label.setText("Sweep stopped")

            # Stop TX if not used for single tone
            if not self.tx_enabled and self.tx_thread is not None:
                self.tx_thread.stop()

                with self.sdr_lock:
                    if self.tx_channel is not None:
                        self.tx_channel.enable = False

                    # Reconfigure back to RX only if scanning
                    if self.live_scanning and self.rx_channel is not None:
                        print("Reconfiguring back to RX only...")
                        time.sleep(0.1)

                        self.sdr.sync_config(
                            layout=_bladerf.ChannelLayout.RX_X1,
                            fmt=_bladerf.Format.SC16_Q11,
                            num_buffers=self.config.SYNC_NUM_BUFFERS,
                            buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                            num_transfers=self.config.SYNC_NUM_TRANSFERS,
                            stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                        )

                        self.rx_channel.enable = True
                        print("RX re-enabled")

            print("Sweep stopped")

    def next_sweep_step(self):
        """Move to next frequency in sweep"""
        if not self.sweep_enabled:
            return

        self.current_sweep_freq += 1
        if self.current_sweep_freq >= len(self.sweep_freqs):
            self.current_sweep_freq = 0

        freq = self.sweep_freqs[self.current_sweep_freq]

        with self.sdr_lock:
            if self.tx_channel is not None:
                self.tx_channel.frequency = int(freq)

        self.sweep_status_label.setText(
            f"TX{self.current_tx_channel_index + 1} Sweep: {freq/1e6:.2f} MHz"
        )

    def on_mouse_moved(self, evt):
        """Handle mouse movement for cursor"""
        pos = evt[0]

        if self.graph_plot.sceneBoundingRect().contains(pos) and \
           self.current_x is not None:
            mouse_point = self.graph_plot.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()

            self.vLine.setPos(x)
            self.hLine.setPos(mouse_point.y())

            # Find nearest point
            idx = (np.abs(self.current_x - x)).argmin()
            freq_val = self.current_x[idx]
            power_val = self.current_y[idx]

            self.cursorLabel.setText(f"{freq_val:.2f} MHz\n{power_val:.1f} dBm")
            self.cursorLabel.setPos(freq_val, power_val)

    def closeEvent(self, event):
        """Handle application close"""
        print("Closing application...")

        # Stop scanning
        self.live_scanning = False

        # Stop TX
        if self.tx_thread is not None:
            self.tx_thread.stop()

        # Stop sweep
        self.sweep_timer.stop()

        # Close device
        with self.sdr_lock:
            # if self.rx_channel is not None:
            #     try:
            #         self.rx_channel.enable = False
            #     except Exception as e:
            #         print(f"Error disabling RX: {e}")

            # if self.tx_channel is not None:
            #     try:
            #         self.tx_channel.enable = False
            #     except Exception as e:
            #         print(f"Error disabling TX: {e}")

            if self.sdr is not None:
                try:
                    self.sdr.close()
                except Exception as e:
                    print(f"Error closing SDR: {e}")

        print("BladeRF closed successfully")
        event.accept()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point"""

    print("=================================================================")
    print("BladeRF 2.0 Wideband Spectrum Analyzer - Starting...")
    print("=================================================================")

    QLocale.setDefault(QLocale("C"))
    print("Qt locale set")

    # Determine base path
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
        print(f"Running as frozen executable from: {base_path}")
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Running as script from: {base_path}")

    # Setup logging
    log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
    log_path = os.path.join(base_path, log_filename)

    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print(f"=================================================================")
    print(f"BladeRF 2.0 Wideband Spectrum Analyzer")
    print(f"Log file: {log_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"=================================================================")

    # Create application
    print("Creating QApplication...")
    app = QApplication(sys.argv)
    print("QApplication created")

    # Set icon
    if getattr(sys, 'frozen', False):
        icon_path = os.path.join(sys._MEIPASS, "bladerf2_0.ico")
    else:
        icon_path = os.path.join(base_path, "bladerf2_0.ico")

    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        print(f"Icon loaded: {icon_path}")
    else:
        print(f"Warning: Icon not found at {icon_path}")

    # Create and show window
    try:
        print("Creating main window...")
        window = SpectrumAnalyzer()
        print("Main window created")

        print("Showing window...")
        window.show()
        print("Window shown - application ready")

        print("Starting event loop...")
        result = app.exec_()
        print(f"Event loop exited with code: {result}")
        sys.exit(result)

    except Exception as e:
        print(f"FATAL ERROR in main: {e}")
        import traceback
        traceback.print_exc()

        # Try to show error dialog
        try:
            QMessageBox.critical(None, "Fatal Error",
                               f"Application failed to start:\n{str(e)}\n\nCheck log file for details.")
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()