"""
BladeRF 2.0 Spectrum Analyzer - Optimized Architecture
With channel selection and logging
"""

import argparse
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
                            QLineEdit, QLabel, QMessageBox, QSpinBox, QFrame,
                            QSlider, QGroupBox, QFileDialog, QScrollArea)
from PyQt5.QtCore import QTimer, QLocale, QThread, QRectF, Qt
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
    step: float = 0.6e6          # 0.53 MHz (equal to sample rate)

    sample_rate: float = step    # 0.53 MHz
    num_samples: int = 4096
    gain: int = 30

    # Waterfall
    waterfall_lines: int = 30

    # BladeRF streaming
    SYNC_NUM_BUFFERS: int = 16
    SYNC_NUM_TRANSFERS: int = 8
    SYNC_STREAM_TIMEOUT: int = 3500
    BUFFER_SIZE_MULTIPLIER: int = 4


# ----------------------------------------------------------------------------
# Absolute calibration table: frequency → dBm offset, linearly interpolated.
# Loaded once from absolute_calibration.csv next to this file. Falls back to a
# flat -77 dB offset (the historical hardcoded value) if the file is missing
# or malformed, so behavior is preserved on a fresh checkout.
# ----------------------------------------------------------------------------

ABS_CAL_FILENAME = "absolute_calibration.csv"

_ABS_CAL_LOADED = False
_ABS_CAL_FREQS = np.array([400e6, 800e6, 1500e6, 2400e6, 3000e6, 5000e6])
_ABS_CAL_OFFSETS = np.array([-77.0, -77.0, -77.0, -77.0, -77.0, -77.0])


def _abs_cal_default_path() -> str:
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, ABS_CAL_FILENAME)


def _load_absolute_calibration():
    global _ABS_CAL_LOADED, _ABS_CAL_FREQS, _ABS_CAL_OFFSETS
    if _ABS_CAL_LOADED:
        return
    _ABS_CAL_LOADED = True
    path = _abs_cal_default_path()
    if not os.path.exists(path):
        print(f"Note: {path} not found — using built-in flat -77 dB offset")
        return
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if data.shape[1] < 2 or len(data) == 0:
            raise ValueError("expected columns: freq_hz, offset_db")
        order = np.argsort(data[:, 0])
        _ABS_CAL_FREQS = data[order, 0]
        _ABS_CAL_OFFSETS = data[order, 1]
        print(f"Absolute calibration loaded: {path} "
              f"({len(_ABS_CAL_FREQS)} points, "
              f"{_ABS_CAL_FREQS.min()/1e6:.0f}–{_ABS_CAL_FREQS.max()/1e6:.0f} MHz)")
    except Exception as e:
        print(f"Warning: failed to parse {path}: {e}. Using flat -77 dB.")


def get_calibration(freq_hz: float) -> float:
    """Absolute-power dBm offset for the given RX frequency (interpolated)."""
    _load_absolute_calibration()
    return float(np.interp(freq_hz, _ABS_CAL_FREQS, _ABS_CAL_OFFSETS))


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
                self.sdr.sync_tx(self.tx_buffer, len(self.tx_buffer) // 4)
                time.sleep(0.001)
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
        self.is_connected = False

        # Current channel indices
        self.current_rx_channel_index = 0
        self.current_tx_channel_index = 0

        # RX state
        self.live_scanning = False
        self.center_freq = 1000e6
        self.buffer = None

        # Composite scanning
        self.wb_centers = None
        self.wb_index = 0
        self.composite_spectrum = None
        self.common_freq = None

        # Max Hold
        self.maxhold_enabled = False
        self.maxhold_data_arr = None

        # Calibration
        self.segment_correction = None   # профиль АЧХ (num_samples,) в дБ
        self.trim_fraction = 0.0
        self.calibrating = False

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

        # IQ recording
        self.iq_recording = False
        self.iq_save_path = None
        self.iq_samples_target = 0
        self.iq_accum = []

        print("State variables initialized")

        try:
            print("Creating UI...")
            self.init_ui()
            print("UI created successfully")
        except Exception as e:
            print(f"ERROR creating UI: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Don't auto-connect - user will click Connect button
        print("Application ready. Click 'Connect' to initialize BladeRF.")

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

    def toggle_connection(self):
        """Connect or disconnect BladeRF"""
        if self.is_connected:
            self.disconnect_bladerf()
        else:
            self.connect_bladerf()

    def connect_bladerf(self):
        """Connect to BladeRF device"""
        try:
            self.connect_button.setEnabled(False)
            self.connection_status_label.setText("Connecting...")
            self.connection_status_label.setStyleSheet("QLabel { color: orange; }")
            QApplication.processEvents()

            self.init_bladerf()

            self.is_connected = True
            self.connect_button.setText("Disconnect BladeRF")
            self.connect_button.setEnabled(True)
            self.connection_status_label.setText("Connected")
            self.connection_status_label.setStyleSheet("QLabel { color: green; }")

            print("BladeRF connected successfully")

        except Exception as e:
            print(f"ERROR connecting to BladeRF: {e}")
            import traceback
            traceback.print_exc()

            self.is_connected = False
            self.connect_button.setText("Connect to BladeRF")
            self.connect_button.setEnabled(True)
            self.connection_status_label.setText("Connection Failed")
            self.connection_status_label.setStyleSheet("QLabel { color: red; }")

            QMessageBox.critical(
                self, "Connection Error",
                f"Failed to connect to BladeRF:\n{str(e)}\n\nPlease check:\n"
                "- Device is connected\n"
                "- USB 3.0 port\n"
                "- udev rules installed\n"
                "- User in plugdev group"
            )

    def disconnect_bladerf(self):
        """Disconnect from BladeRF device"""
        try:
            # Stop all active operations
            if self.live_scanning:
                self.live_scanning = False
                self.scan_button.setText("Start Scanning")

            if self.tx_enabled:
                self.tx_enabled = False
                if self.tx_thread is not None:
                    self.tx_thread.stop()
                self.tx_start_button.setText("Start Transmission")
                self.tx_status_label.setText("")

            if self.sweep_enabled:
                self.sweep_enabled = False
                self.sweep_timer.stop()
                self.sweep_start_button.setText("Start Sweep")
                self.sweep_status_label.setText("")

            if self.iq_recording:
                self.iq_recording = False
                self.iq_record_button.setEnabled(True)
                self.iq_status_label.setText("Ready")

            # Close device
            with self.sdr_lock:
                if self.rx_channel is not None:
                    try:
                        self.rx_channel.enable = False
                    except:
                        pass
                    self.rx_channel = None

                if self.tx_channel is not None:
                    try:
                        self.tx_channel.enable = False
                    except:
                        pass
                    self.tx_channel = None

                if self.sdr is not None:
                    try:
                        self.sdr.close()
                    except:
                        pass
                    self.sdr = None

            self.is_connected = False
            self.connect_button.setText("Connect to BladeRF")
            self.connection_status_label.setText("Disconnected")
            self.connection_status_label.setStyleSheet("QLabel { color: red; }")

            print("BladeRF disconnected")

        except Exception as e:
            print(f"Error during disconnect: {e}")
            import traceback
            traceback.print_exc()

    def init_ui(self):
        """Initialize user interface"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ===== LEFT: Graphs =====
        graph_container = QWidget()
        graph_layout = QVBoxLayout(graph_container)

        self.graph_plot = pg.PlotWidget(title="Spectrum in dBm")
        self.graph_plot.setLabel('left', 'Power', units='dBm')
        self.graph_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.graph_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.graph_plot.setYRange(-140, -30)
        self.graph_plot.showGrid(x=True, y=True)

        self.curve = self.graph_plot.plot(pen='y')
        self.maxhold_curve = self.graph_plot.plot(pen='r')

        self.max_marker = self.graph_plot.plot(
            symbol='o', symbolBrush='r', symbolSize=10
        )
        self.max_text = pg.TextItem(color='w', anchor=(0, 1))
        self.graph_plot.addItem(self.max_text)

        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('c'))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('c'))
        self.graph_plot.addItem(self.vLine, ignoreBounds=True)
        self.graph_plot.addItem(self.hLine, ignoreBounds=True)
        self.cursorLabel = pg.TextItem("", anchor=(1, 1),
                                       fill=pg.mkBrush(0, 0, 0, 150))
        self.graph_plot.addItem(self.cursorLabel)

        self.proxy = pg.SignalProxy(
            self.graph_plot.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.on_mouse_moved
        )

        graph_layout.addWidget(self.graph_plot)

        self.waterfall_plot = pg.PlotWidget(title="Waterfall")
        self.waterfall_plot.setLabel('bottom', 'Frequency', units='MHz')
        self.waterfall_plot.setLabel('left', 'Scan')
        self.waterfall_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.waterfall_img = pg.ImageItem()
        self.waterfall_img.setOpts(invertY=False, axisOrder='row-major')

        lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        self.waterfall_img.setLookupTable(lut)
        self.waterfall_plot.addItem(self.waterfall_img)
        self.waterfall_plot.getViewBox().invertY(True)

        graph_layout.addWidget(self.waterfall_plot)

        # ===== Waterfall color range sliders =====
        wf_controls = QGroupBox("Waterfall Color Range")
        wf_controls_layout = QHBoxLayout(wf_controls)

        # Min slider
        wf_controls_layout.addWidget(QLabel("Min:"))
        self.wf_min_label = QLabel("-140 dBm")
        self.wf_min_label.setMinimumWidth(70)
        self.wf_min_slider = QSlider(Qt.Horizontal)
        self.wf_min_slider.setRange(-180, 0)
        self.wf_min_slider.setValue(-140)
        self.wf_min_slider.setTickPosition(QSlider.TicksBelow)
        self.wf_min_slider.setTickInterval(20)
        self.wf_min_slider.valueChanged.connect(self.on_wf_range_changed)
        wf_controls_layout.addWidget(self.wf_min_slider, stretch=1)
        wf_controls_layout.addWidget(self.wf_min_label)

        wf_controls_layout.addSpacing(20)

        # Max slider
        wf_controls_layout.addWidget(QLabel("Max:"))
        self.wf_max_label = QLabel("-30 dBm")
        self.wf_max_label.setMinimumWidth(70)
        self.wf_max_slider = QSlider(Qt.Horizontal)
        self.wf_max_slider.setRange(-180, 0)
        self.wf_max_slider.setValue(-30)
        self.wf_max_slider.setTickPosition(QSlider.TicksBelow)
        self.wf_max_slider.setTickInterval(20)
        self.wf_max_slider.valueChanged.connect(self.on_wf_range_changed)
        wf_controls_layout.addWidget(self.wf_max_slider, stretch=1)
        wf_controls_layout.addWidget(self.wf_max_label)

        # Reset button
        wf_reset_btn = QPushButton("Reset")
        wf_reset_btn.setMaximumWidth(60)
        wf_reset_btn.clicked.connect(self.reset_wf_range)
        wf_controls_layout.addWidget(wf_reset_btn)

        graph_layout.addWidget(wf_controls)
        main_layout.addWidget(graph_container, stretch=3)

        # ===== RIGHT: Control Panel =====
        control_panel = QWidget()
        control_layout = QFormLayout(control_panel)

        # Connection control
        control_layout.addRow(QLabel("<b>Device Connection</b>"))

        self.connect_button = QPushButton("Connect to BladeRF")
        self.connect_button.setStyleSheet("QPushButton { font-weight: bold; min-height: 30px; }")
        self.connect_button.clicked.connect(self.toggle_connection)
        control_layout.addRow(self.connect_button)

        self.connection_status_label = QLabel("Disconnected")
        self.connection_status_label.setStyleSheet("QLabel { color: red; }")
        control_layout.addRow("Status:", self.connection_status_label)

        # Separator
        separator0 = QFrame()
        separator0.setFrameShape(QFrame.HLine)
        separator0.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(separator0)

        control_layout.addRow(QLabel("<b>Receiver Settings</b>"))

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

        # --- Calibration ---
        self.calibrate_button = QPushButton("Calibrate Profile")
        self.calibrate_button.clicked.connect(self.run_calibration)
        control_layout.addRow(self.calibrate_button)

        self.calibrate_status_label = QLabel("Not calibrated")
        control_layout.addRow("Cal status:", self.calibrate_status_label)

        self.calibrate_clear_button = QPushButton("Clear Calibration")
        self.calibrate_clear_button.clicked.connect(self.clear_calibration)
        control_layout.addRow(self.calibrate_clear_button)

        self.calibrate_save_button = QPushButton("Save Calibration…")
        self.calibrate_save_button.clicked.connect(self.on_save_calibration)
        control_layout.addRow(self.calibrate_save_button)

        self.calibrate_load_button = QPushButton("Load Calibration…")
        self.calibrate_load_button.clicked.connect(self.on_load_calibration)
        control_layout.addRow(self.calibrate_load_button)

        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(separator1)

        # TX Settings
        control_layout.addRow(QLabel("<b>Transmitter Settings</b>"))

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

        # Separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.HLine)
        separator3.setFrameShadow(QFrame.Sunken)
        control_layout.addRow(separator3)

        # IQ Save
        control_layout.addRow(QLabel("<b>IQ Data Recording</b>"))

        self.iq_freq_edit = QLineEdit("2400")
        control_layout.addRow("Center Freq (MHz):", self.iq_freq_edit)

        self.iq_duration_spin = QSpinBox()
        self.iq_duration_spin.setRange(1, 60000)
        self.iq_duration_spin.setValue(1000)
        self.iq_duration_spin.setSuffix(" ms")
        control_layout.addRow("Duration:", self.iq_duration_spin)

        self.iq_record_button = QPushButton("Record && Save IQ...")
        self.iq_record_button.clicked.connect(self.start_iq_recording)
        control_layout.addRow(self.iq_record_button)

        self.iq_status_label = QLabel("Ready")
        control_layout.addRow("IQ Status:", self.iq_status_label)

        # Wrap control panel in scroll area
        scroll = QScrollArea()
        scroll.setWidget(control_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        main_layout.addWidget(scroll, stretch=1)

    def init_bladerf(self):
        """Initialize BladeRF device"""
        try:
            print("Opening BladeRF device...")
            with self.sdr_lock:
                self.sdr = _bladerf.BladeRF()
                print(f"BladeRF device opened: {self.sdr.device_speed}")

                print("Initializing RX channel...")
                self.init_rx_channel(self.current_rx_channel_index)
                print("RX channel initialized")

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

        if self.rx_channel is not None:
            try:
                print("Disabling old RX channel...")
                self.rx_channel.enable = False
                print("Old RX channel disabled")
            except Exception as e:
                print(f"Warning disabling old RX channel: {e}")
            time.sleep(0.2)

        print(f"Creating RX channel object for index {channel_index}...")
        try:
            rx_ch = _bladerf.CHANNEL_RX(channel_index)
            print(f"CHANNEL_RX enum created: {rx_ch}")
            self.rx_channel = self.sdr.Channel(rx_ch)
            print("RX channel object created successfully")
        except Exception as e:
            print(f"ERROR creating channel with Method 1: {e}")
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

        print("Setting RX sample rate...")
        self.rx_channel.sample_rate = int(self.config.sample_rate)
        print(f"  Sample rate set: {self.config.sample_rate}")

        print("Setting RX bandwidth...")
        self.rx_channel.bandwidth = int(self.config.sample_rate)

        print("Setting RX gain mode...")
        self.rx_channel.gain_mode = _bladerf.GainMode.Manual

        print("Setting RX gain...")
        self.rx_channel.gain = self.config.gain

        print("Setting RX frequency...")
        self.rx_channel.frequency = int(self.center_freq)

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
                if self.tx_channel is not None:
                    try:
                        self.tx_channel.enable = False
                    except Exception as e:
                        print(f"Warning disabling old TX channel: {e}")
                    time.sleep(0.2)

                self.tx_channel = self.sdr.Channel(_bladerf.CHANNEL_TX(channel_index))
                self.tx_channel.sample_rate = int(self.config.sample_rate)
                self.tx_channel.bandwidth = int(self.config.sample_rate)
                self.tx_channel.gain_mode = _bladerf.GainMode.Manual
                self.tx_channel.gain = self.tx_gain_spin.value()

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
        if not self.is_connected:
            return

        if self.sdr is None:
            return

        was_scanning = self.live_scanning
        if was_scanning:
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")

        try:
            self.init_rx_channel(index)
            print(f"Switched to RX{index + 1}")

            if was_scanning:
                QTimer.singleShot(300, lambda: self.scan_button.click())

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch RX channel: {e}")

    def on_tx_channel_changed(self, index: int):
        """Handle TX channel change"""
        if not self.is_connected:
            return

        if self.sdr is None:
            return

        was_transmitting = self.tx_enabled
        was_sweeping = self.sweep_enabled

        if was_transmitting:
            self.start_transmission()
        if was_sweeping:
            self.toggle_sweep_transmission()

        try:
            self.init_tx_channel(index)
            print(f"Switched to TX{index + 1}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch TX channel: {e}")

    def acquire_one_spectrum(self) -> tuple:
        """Acquire one spectrum. Applies AЧХ correction if calibrated."""
        with self.sdr_lock:
            try:
                buf = bytearray(self.config.num_samples * 4)
                self.sdr.sync_rx(buf, self.config.num_samples)

                samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
                samples = samples.view(np.complex64)
                samples /= 2048.0

                window = np.blackman(len(samples))
                window_power = np.sum(window ** 2)

                spectrum = np.fft.fftshift(np.fft.fft(samples * window))
                power = np.abs(spectrum) ** 2 / window_power

                cal_offset = get_calibration(self.center_freq)
                power_dbm = 10 * np.log10(power / 1e-3 + 1e-12) + cal_offset

                # Применяем коррекцию профиля АЧХ если откалиброван
                if (self.segment_correction is not None and
                        len(self.segment_correction) == len(power_dbm)):
                    power_dbm = power_dbm - self.segment_correction

                freqs = np.fft.fftshift(
                    np.fft.fftfreq(self.config.num_samples,
                                   d=1.0 / self.config.sample_rate)
                )
                freq_axis = (freqs + self.center_freq) / 1e6

                return freq_axis, power_dbm

            except Exception as e:
                print(f"Error acquiring spectrum: {e}")
                freq_axis = np.linspace(0, 1, self.config.num_samples)
                return freq_axis, np.full(self.config.num_samples, -140.0)

    # =========================================================================
    # Calibration
    # =========================================================================

    def run_calibration(self, averages: int = 200):
        """
        Average N spectra at the current frequency → AЧХ (filter-shape) profile.
        Profile is normalized so the middle third of the segment = 0 dB.
        From then on each segment gets corrected against this profile.
        """
        if not self.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to BladeRF first")
            return

        if self.sdr is None:
            QMessageBox.warning(self, "Error", "BladeRF not initialized")
            return

        if self.live_scanning:
            QMessageBox.warning(self, "Error",
                                "Stop scanning before calibration")
            return

        self.calibrating = True
        self.calibrate_button.setEnabled(False)
        self.calibrate_status_label.setText("Calibrating...")
        QApplication.processEvents()

        CAL_AVERAGES = int(averages)

        try:
            # Сначала сбрасываем буфер
            flush_buf = bytearray(self.config.num_samples * 4)
            flush_count = (self.config.SYNC_NUM_BUFFERS *
                           self.config.BUFFER_SIZE_MULTIPLIER + 4)
            with self.sdr_lock:
                for _ in range(flush_count):
                    try:
                        self.sdr.sync_rx(flush_buf, self.config.num_samples)
                    except Exception:
                        break

            # Накапливаем в линейной шкале для корректного усреднения
            accum = np.zeros(self.config.num_samples, dtype=np.float64)

            for i in range(CAL_AVERAGES):
                _, power_dbm = self.acquire_one_spectrum()
                # Переводим в линейный масштаб перед суммированием
                accum += 10.0 ** (power_dbm / 10.0)

                if i % 20 == 0:
                    self.calibrate_status_label.setText(
                        f"Calibrating... {i}/{CAL_AVERAGES}")
                    QApplication.processEvents()

            # Средний профиль в дБ
            avg_profile = 10.0 * np.log10(accum / CAL_AVERAGES)

            # Опорный уровень = медиана центральной трети (там фильтр ровный)
            n = len(avg_profile)
            center_slice = slice(n // 3, 2 * n // 3)
            ref_level = np.median(avg_profile[center_slice])

            # Коррекция = сколько надо вычесть из каждого бина
            # Положительная коррекция опускает бин, отрицательная поднимает
            self.segment_correction = avg_profile - ref_level

            peak_correction = max(abs(self.segment_correction.min()),
                                  abs(self.segment_correction.max()))
            edge_left  = self.segment_correction[:n // 8].mean()
            edge_right = self.segment_correction[-n // 8:].mean()

            print(f"Calibration done.")
            print(f"  Ref level: {ref_level:.1f} dBm")
            print(f"  Profile range: {self.segment_correction.min():.2f} .. "
                  f"{self.segment_correction.max():.2f} dB")
            print(f"  Edge correction left={edge_left:.2f} dB  "
                  f"right={edge_right:.2f} dB")

            self.calibrate_status_label.setText(
                f"OK  peak±{peak_correction:.1f} dB  "
                f"L{edge_left:+.1f} R{edge_right:+.1f} dB"
            )

        except Exception as e:
            print(f"Calibration error: {e}")
            import traceback
            traceback.print_exc()
            self.segment_correction = None
            self.calibrate_status_label.setText(f"Error: {e}")

        finally:
            self.calibrating = False
            self.calibrate_button.setEnabled(True)

    def clear_calibration(self):
        """Remove АЧХ correction"""
        self.segment_correction = None
        self.calibrate_status_label.setText("Not calibrated")
        print("Calibration cleared")

    def save_calibration_to_file(self, filepath: str):
        """Persist the current segment-correction profile to a .npz file.

        Stores the FFT size, sample rate, center frequency, gain and timestamp
        alongside the array so load can verify compatibility.
        """
        if self.segment_correction is None:
            raise ValueError("No calibration to save — run calibration first")
        np.savez(
            filepath,
            correction=self.segment_correction,
            sample_rate=float(self.config.sample_rate),
            fft_size=int(self.config.num_samples),
            center_freq=float(self.center_freq),
            gain=int(self.config.gain),
            timestamp=datetime.now().isoformat(),
        )
        print(f"Calibration saved: {filepath} "
              f"(FFT={self.config.num_samples}, "
              f"SR={self.config.sample_rate/1e6:.3f} MHz)")

    def load_calibration_from_file(self, filepath: str) -> tuple:
        """Load a .npz calibration. Returns (ok: bool, message: str)."""
        try:
            data = np.load(filepath)
            correction = data["correction"]
            saved_sr = float(data["sample_rate"])
            saved_fft = int(data["fft_size"])
        except Exception as e:
            return False, f"Load failed: {e}"

        if saved_fft != self.config.num_samples:
            return False, (f"FFT size mismatch: file has {saved_fft}, "
                           f"current is {self.config.num_samples}")

        self.segment_correction = correction
        sr_match = abs(saved_sr - self.config.sample_rate) < 1.0
        if sr_match:
            msg = (f"Loaded ({saved_fft} bins @ "
                   f"{saved_sr/1e6:.3f} MHz SR)")
        else:
            msg = (f"Loaded — WARNING: SR mismatch (file "
                   f"{saved_sr/1e6:.3f} vs current "
                   f"{self.config.sample_rate/1e6:.3f} MHz)")
        print(f"Calibration loaded from {filepath}: {msg}")
        return True, msg

    def on_save_calibration(self):
        """GUI handler — Save Calibration button"""
        if self.segment_correction is None:
            QMessageBox.warning(self, "Error",
                                "No calibration data — run calibration first")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", os.path.expanduser("~"),
            "Calibration files (*.npz)")
        if not filepath:
            return
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        try:
            self.save_calibration_to_file(filepath)
            self.calibrate_status_label.setText(
                f"Saved {os.path.basename(filepath)}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Save failed: {e}")

    def on_load_calibration(self):
        """GUI handler — Load Calibration button"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", os.path.expanduser("~"),
            "Calibration files (*.npz)")
        if not filepath:
            return
        ok, msg = self.load_calibration_from_file(filepath)
        if ok:
            self.calibrate_status_label.setText(msg)
        else:
            QMessageBox.warning(self, "Error", msg)
            self.calibrate_status_label.setText("Load failed")

    # =========================================================================
    # Scanning
    # =========================================================================

    def toggle_live_scanning(self):
        """Toggle live scanning mode"""
        if not self.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to BladeRF first")
            return

        if self.live_scanning:
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
        else:
            try:
                self.config.num_samples = int(self.samples_combo.currentText())
                self.config.waterfall_lines = self.waterfall_lines_spin.value()
                self.config.gain = self.gain_spin.value()

                with self.sdr_lock:
                    self.rx_channel.enable = False
                    time.sleep(0.1)
                    self.rx_channel.gain = self.config.gain
                    self.rx_channel.enable = True
                    time.sleep(0.05)

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
            stop_mhz  = float(self.stop_freq_edit.text())
            step_mhz  = float(self.step_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid frequency values")
            self.live_scanning = False
            self.scan_button.setText("Start Scanning")
            return

        BLADERF_MAX_SR = 61.44e6
        BLADERF_MIN_SR = 0.521e6
        TRIM = 0.15

        desired_sr = step_mhz * 1e6 / (1.0 - 2.0 * TRIM) * 1.05
        desired_sr = float(np.clip(desired_sr, BLADERF_MIN_SR, BLADERF_MAX_SR))

        with self.sdr_lock:
            self.rx_channel.sample_rate = int(desired_sr)
            self.rx_channel.bandwidth   = int(desired_sr)
            # Читаем реально установленное железом значение
            actual_sr = float(self.rx_channel.sample_rate)
            self.config.sample_rate = actual_sr

            # Переконфигурируем стриминг под новый SR
            self.sdr.sync_config(
                layout=_bladerf.ChannelLayout.RX_X1,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=self.config.SYNC_NUM_BUFFERS,
                buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                num_transfers=self.config.SYNC_NUM_TRANSFERS,
                stream_timeout=self.config.SYNC_STREAM_TIMEOUT
            )
            self.rx_channel.enable = True

        # Пересчитываем trim от реального SR
        if actual_sr > step_mhz * 1e6:
            self.trim_fraction = 0.5 * (1.0 - (step_mhz * 1e6) / actual_sr) * 0.95
        else:
            self.trim_fraction = 0.0

        print(f"Step={step_mhz} MHz | Requested SR={desired_sr/1e6:.3f} MHz | "
              f"Actual SR={actual_sr/1e6:.3f} MHz | trim={self.trim_fraction:.4f}")

        start_hz = start_mhz * 1e6
        stop_hz  = stop_mhz  * 1e6
        step_hz  = step_mhz  * 1e6

        # Центры строго по шагу, не зависят от SR
        self.wb_centers = np.arange(
            start_hz + step_hz / 2,
            stop_hz,
            step_hz
        )
        print(f"Centers (MHz): {[f'{c/1e6:.1f}' for c in self.wb_centers]}")

        self.wb_index = 0

        common_res = 0.1
        num_points  = int(np.round((stop_mhz - start_mhz) / common_res)) + 1
        self.common_freq = np.linspace(start_mhz, stop_mhz, num_points)
        self.composite_spectrum = np.full(self.common_freq.shape, -140.0)

        self.segment_accum   = None
        self.segment_count   = 0
        self.segment_avg_len = 20

        self.waterfall_img.setRect(
            QRectF(start_mhz, 0, stop_mhz - start_mhz, self.config.waterfall_lines)
        )
        self.curve.clear()
        self.waterfall_data.fill(-140)
        self.waterfall_index = 0

        # Калибровка недействительна если изменился SR или num_samples
        if self.segment_correction is not None:
            if len(self.segment_correction) != self.config.num_samples:
                self.segment_correction = None
                self.calibrate_status_label.setText(
                    "Recalibrate needed (FFT size changed)")
                print("Warning: calibration reset — FFT size changed")
            else:
                print("Existing calibration retained (SR/trim changed, "
                      "but profile shape is still valid)")

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

            # Полный сброс буфера после перестройки частоты
            flush_count = (self.config.SYNC_NUM_BUFFERS *
                           self.config.BUFFER_SIZE_MULTIPLIER + 4)
            flush_buf = bytearray(self.config.num_samples * 4)
            for _ in range(flush_count):
                try:
                    self.sdr.sync_rx(flush_buf, self.config.num_samples)
                except Exception:
                    break

        self.segment_accum = None
        self.segment_count = 0

        QTimer.singleShot(0, self.do_composite_measurement)

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

            if self.segment_count < self.segment_avg_len:
                QTimer.singleShot(0, self.do_composite_measurement)
                return

            avg_power = self.segment_accum / self.segment_count

            # Обрезаем края сегмента
            trim = int(len(meas_freq) * self.trim_fraction)
            if trim > 0:
                meas_freq_use = meas_freq[trim:-trim]
                avg_power_use = avg_power[trim:-trim]
            else:
                meas_freq_use = meas_freq
                avg_power_use = avg_power

            seg_min = meas_freq_use[0]
            seg_max = meas_freq_use[-1]
            mask = (self.common_freq >= seg_min) & (self.common_freq <= seg_max)

            if np.any(mask):
                interp_power = np.interp(
                    self.common_freq[mask],
                    meas_freq_use,
                    avg_power_use,
                    left=-140,
                    right=-140
                )

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
                levels=(self.wf_min_slider.value(), self.wf_max_slider.value())
            )

        except Exception as e:
            print(f"Display error: {e}")

    def on_wf_range_changed(self):
        """Update waterfall color levels from sliders"""
        wf_min = self.wf_min_slider.value()
        wf_max = self.wf_max_slider.value()

        # Не даём min >= max
        if wf_min >= wf_max:
            if self.sender() == self.wf_min_slider:
                wf_min = wf_max - 1
                self.wf_min_slider.blockSignals(True)
                self.wf_min_slider.setValue(wf_min)
                self.wf_min_slider.blockSignals(False)
            else:
                wf_max = wf_min + 1
                self.wf_max_slider.blockSignals(True)
                self.wf_max_slider.setValue(wf_max)
                self.wf_max_slider.blockSignals(False)

        self.wf_min_label.setText(f"{wf_min} dBm")
        self.wf_max_label.setText(f"{wf_max} dBm")

        if self.waterfall_data is not None:
            self.waterfall_img.setImage(
                self.waterfall_data,
                autoLevels=False,
                levels=(wf_min, wf_max)
            )

    def reset_wf_range(self):
        """Reset waterfall range to defaults"""
        self.wf_min_slider.setValue(-140)
        self.wf_max_slider.setValue(-30)

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
        if not self.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to BladeRF first")
            return

        if not self.tx_enabled:
            try:
                tx_freq = float(self.tx_freq_edit.text()) * 1e6
                tx_gain = self.tx_gain_spin.value()

                if self.tx_channel is None:
                    self.init_tx_channel(self.current_tx_channel_index)

                print("Configuring for TX transmission...")
                with self.sdr_lock:
                    if self.tx_channel is not None:
                        self.tx_channel.enable = False

                    time.sleep(0.1)

                    print("Configuring TX streaming...")
                    self.sdr.sync_config(
                        layout=_bladerf.ChannelLayout.TX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=self.config.SYNC_NUM_BUFFERS,
                        buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                        num_transfers=self.config.SYNC_NUM_TRANSFERS,
                        stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                    )

                    self.tx_channel.frequency = int(tx_freq)
                    self.tx_channel.gain = tx_gain
                    self.tx_channel.enable = True
                    time.sleep(0.05)

                t = np.arange(self.config.num_samples)
                tone_freq = 1000
                signal = np.exp(1j * 2 * np.pi * tone_freq * t / self.config.sample_rate)

                iq = np.empty(2 * len(signal), dtype=np.int16)
                iq[0::2] = np.clip(np.real(signal) * 2047, -2048, 2047).astype(np.int16)
                iq[1::2] = np.clip(np.imag(signal) * 2047, -2048, 2047).astype(np.int16)
                self.tx_buffer = iq.tobytes()

                self.tx_thread = TXThread(self.sdr, self.tx_buffer)
                self.tx_thread.start()

                self.tx_enabled = True
                self.tx_start_button.setText("Stop Transmission")
                self.tx_status_label.setText(
                    f"TX{self.current_tx_channel_index + 1}: {tx_freq/1e6:.2f} MHz, Gain {tx_gain}"
                )

                print(f"TX started on channel {self.current_tx_channel_index + 1} "
                      f"at {tx_freq/1e6:.2f} MHz")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to start TX: {e}")
                import traceback
                traceback.print_exc()
                self.tx_enabled = False
        else:
            self.tx_enabled = False

            if self.tx_thread is not None:
                self.tx_thread.stop()

            with self.sdr_lock:
                if self.tx_channel is not None:
                    self.tx_channel.enable = False

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
        if not self.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to BladeRF first")
            return

        if not self.sweep_enabled:
            try:
                start_freq = float(self.sweep_start_edit.text()) * 1e6
                stop_freq  = float(self.sweep_stop_edit.text()) * 1e6
                step_freq  = float(self.sweep_step_edit.text()) * 1e6
                dwell_time = float(self.sweep_dwell_edit.text())

                if step_freq <= 0:
                    raise ValueError("Step must be positive")

                if start_freq < stop_freq:
                    self.sweep_freqs = np.arange(start_freq, stop_freq + step_freq, step_freq)
                else:
                    self.sweep_freqs = np.arange(start_freq, stop_freq - step_freq, -step_freq)

                if len(self.sweep_freqs) == 0:
                    raise ValueError("Invalid sweep parameters")

                if self.tx_channel is None:
                    self.init_tx_channel(self.current_tx_channel_index)

                tx_gain = self.sweep_gain_spin.value()

                print("Configuring for TX sweep...")
                with self.sdr_lock:
                    if self.tx_channel is not None:
                        self.tx_channel.enable = False

                    time.sleep(0.1)

                    self.sdr.sync_config(
                        layout=_bladerf.ChannelLayout.TX_X1,
                        fmt=_bladerf.Format.SC16_Q11,
                        num_buffers=self.config.SYNC_NUM_BUFFERS,
                        buffer_size=self.config.num_samples * self.config.BUFFER_SIZE_MULTIPLIER,
                        num_transfers=self.config.SYNC_NUM_TRANSFERS,
                        stream_timeout=self.config.SYNC_STREAM_TIMEOUT
                    )

                    self.tx_channel.gain = tx_gain
                    self.tx_channel.enable = True
                    time.sleep(0.05)

                t = np.arange(self.config.num_samples)
                tone_freq = 1000
                signal = np.exp(1j * 2 * np.pi * tone_freq * t / self.config.sample_rate)

                iq = np.empty(2 * len(signal), dtype=np.int16)
                iq[0::2] = np.clip(np.real(signal) * 2047, -2048, 2047).astype(np.int16)
                iq[1::2] = np.clip(np.imag(signal) * 2047, -2048, 2047).astype(np.int16)
                self.tx_buffer = iq.tobytes()

                if self.tx_thread is None or not self.tx_thread.isRunning():
                    self.tx_thread = TXThread(self.sdr, self.tx_buffer)
                    self.tx_thread.start()

                self.sweep_enabled = True
                self.current_sweep_freq = 0
                self.sweep_timer.start(int(dwell_time))

                self.sweep_start_button.setText("Stop Sweep")
                self.sweep_status_label.setText(
                    f"TX{self.current_tx_channel_index + 1} Sweep: "
                    f"{self.sweep_freqs[0]/1e6:.2f} MHz"
                )

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
            self.sweep_enabled = False
            self.sweep_timer.stop()

            self.sweep_start_button.setText("Start Sweep")
            self.sweep_status_label.setText("Sweep stopped")

            if not self.tx_enabled and self.tx_thread is not None:
                self.tx_thread.stop()

                with self.sdr_lock:
                    if self.tx_channel is not None:
                        self.tx_channel.enable = False

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

            idx = (np.abs(self.current_x - x)).argmin()
            freq_val  = self.current_x[idx]
            power_val = self.current_y[idx]

            self.cursorLabel.setText(f"{freq_val:.2f} MHz\n{power_val:.1f} dBm")
            self.cursorLabel.setPos(freq_val, power_val)

    def start_iq_recording(self):
        """Ask user for file path, then record IQ samples to .bin / .csv / .npy"""
        if not self.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to BladeRF first")
            return

        if self.sdr is None:
            QMessageBox.warning(self, "Error", "BladeRF not initialized")
            return

        if self.iq_recording:
            QMessageBox.warning(self, "Error", "Recording already in progress")
            return

        # Диалог выбора файла
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save IQ Data",
            os.path.expanduser("~"),
            "NumPy binary (*.npy);;Raw int16 IQ binary (*.bin);;CSV complex (*.csv)"
        )

        if not filepath:
            return  # пользователь отменил

        # Определяем формат по расширению / фильтру
        if filepath.endswith(".npy"):
            fmt = "npy"
        elif filepath.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "bin"
            if not filepath.endswith(".bin"):
                filepath += ".bin"

        try:
            center_hz = float(self.iq_freq_edit.text()) * 1e6
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid center frequency")
            return

        self._iq_begin_recording(filepath, fmt, center_hz,
                                 self.iq_duration_spin.value())

    def _iq_begin_recording(self, filepath: str, fmt: str,
                            center_hz: float, duration_ms: int):
        """Start an IQ capture given an explicit destination + parameters.

        Separated from start_iq_recording so headless mode can drive it
        without a QFileDialog.
        """
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        reads_needed = max(1, int(np.ceil(num_samples / self.config.num_samples)))
        total_samples = reads_needed * self.config.num_samples

        print(f"IQ recording: {center_hz/1e6:.3f} MHz, {duration_ms} ms, "
              f"{total_samples} samples, format={fmt}, file={filepath}")

        self.iq_save_path = filepath
        self.iq_fmt = fmt
        self.iq_center_hz = center_hz
        self.iq_reads_needed = reads_needed
        self.iq_reads_done = 0
        self.iq_accum = []
        self.iq_recording = True

        self.iq_record_button.setEnabled(False)
        self.iq_status_label.setText("Recording...")

        was_scanning = self.live_scanning
        if was_scanning:
            self.live_scanning = False

        with self.sdr_lock:
            self.rx_channel.frequency = int(center_hz)
            self.center_freq = center_hz

            flush_count = (self.config.SYNC_NUM_BUFFERS *
                           self.config.BUFFER_SIZE_MULTIPLIER + 4)
            flush_buf = bytearray(self.config.num_samples * 4)
            for _ in range(flush_count):
                try:
                    self.sdr.sync_rx(flush_buf, self.config.num_samples)
                except Exception:
                    break

        self._iq_was_scanning = was_scanning
        QTimer.singleShot(0, self._iq_read_chunk)

    def _iq_read_chunk(self):
        """Read one chunk of IQ samples during recording"""
        if not self.iq_recording:
            return

        try:
            with self.sdr_lock:
                buf = bytearray(self.config.num_samples * 4)
                self.sdr.sync_rx(buf, self.config.num_samples)

            samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
            samples = samples.view(np.complex64)
            samples /= 2048.0
            self.iq_accum.append(samples.copy())
            self.iq_reads_done += 1

            progress = int(self.iq_reads_done / self.iq_reads_needed * 100)
            self.iq_status_label.setText(f"Recording... {progress}%")

            if self.iq_reads_done < self.iq_reads_needed:
                QTimer.singleShot(0, self._iq_read_chunk)
            else:
                self._iq_save()

        except Exception as e:
            print(f"IQ read error: {e}")
            import traceback
            traceback.print_exc()
            self.iq_recording = False
            self.iq_record_button.setEnabled(True)
            self.iq_status_label.setText(f"Error: {e}")

    def _iq_save(self):
        """Save accumulated IQ data to file"""
        try:
            all_samples = np.concatenate(self.iq_accum)
            filepath = self.iq_save_path
            fmt = self.iq_fmt

            if fmt == "npy":
                # complex64, легко читается: np.load(filepath)
                np.save(filepath, all_samples)

            elif fmt == "csv":
                # Два столбца: real, imag
                data = np.column_stack([all_samples.real, all_samples.imag])
                np.savetxt(filepath, data, delimiter=",",
                           header="real,imag", comments="")

            else:  # bin
                # Interleaved int16: I0 Q0 I1 Q1 ...
                iq_int16 = np.empty(len(all_samples) * 2, dtype=np.int16)
                iq_int16[0::2] = np.clip(all_samples.real * 2047, -2048, 2047).astype(np.int16)
                iq_int16[1::2] = np.clip(all_samples.imag * 2047, -2048, 2047).astype(np.int16)
                iq_int16.tofile(filepath)

            # Сохраняем метаданные в текстовый файл рядом
            meta_path = filepath.rsplit(".", 1)[0] + "_meta.txt"
            with open(meta_path, "w") as f:
                f.write(f"center_freq_hz={self.iq_center_hz}\n")
                f.write(f"sample_rate_hz={self.config.sample_rate}\n")
                f.write(f"num_samples={len(all_samples)}\n")
                f.write(f"format={fmt}\n")
                f.write(f"gain={self.config.gain}\n")
                f.write(f"recorded_at={datetime.now().isoformat()}\n")

            size_kb = os.path.getsize(filepath) / 1024
            print(f"IQ saved: {filepath}  ({len(all_samples)} samples, {size_kb:.1f} KB)")
            print(f"Metadata: {meta_path}")

            self.iq_status_label.setText(
                f"Saved {len(all_samples)} samples  ({size_kb:.0f} KB)"
            )

        except Exception as e:
            print(f"IQ save error: {e}")
            import traceback
            traceback.print_exc()
            self.iq_status_label.setText(f"Save error: {e}")

        finally:
            self.iq_recording = False
            self.iq_accum = []
            self.iq_record_button.setEnabled(True)

            # Возобновляем сканирование если было активно
            if self._iq_was_scanning:
                QTimer.singleShot(200, lambda: self.scan_button.click())

    def closeEvent(self, event):
        """Handle application close"""
        print("Closing application...")

        if self.is_connected:
            self.disconnect_bladerf()

        print("Application closed successfully")
        event.accept()


# ============================================================================
# Headless (CLI) Mode
# ============================================================================

def _iq_fmt_from_path(path: str) -> tuple:
    """Return (filepath, fmt) — coerce filepath to have correct extension."""
    if path.endswith(".npy"):
        return path, "npy"
    if path.endswith(".csv"):
        return path, "csv"
    if not path.endswith(".bin"):
        path += ".bin"
    return path, "bin"


def run_headless(args) -> int:
    """Run the analyzer without showing a window.

    Reuses SpectrumAnalyzer's device-control methods, driving them
    through QTimer callbacks instead of UI events. Returns process exit code.
    """
    # Default to offscreen Qt platform so this works over SSH / in CI
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    QLocale.setDefault(QLocale("C"))
    app = QApplication(sys.argv)

    analyzer = SpectrumAnalyzer()
    # Intentionally NOT calling analyzer.show()

    exit_code = {"value": 0}

    def finish(code: int = 0):
        exit_code["value"] = code
        try:
            if analyzer.is_connected:
                analyzer.disconnect_bladerf()
        except Exception as e:
            print(f"Warning during disconnect: {e}")
        app.quit()

    def fail(msg: str, code: int = 1):
        print(f"ERROR: {msg}", file=sys.stderr)
        finish(code)

    def do_connect() -> bool:
        try:
            analyzer.connect_bladerf()
        except Exception as e:
            fail(f"Connect failed: {e}")
            return False
        if not analyzer.is_connected:
            fail("Connect did not succeed")
            return False
        return True

    def run_info():
        if not do_connect():
            return
        sdr = analyzer.sdr
        print("=== BladeRF Device Info ===")
        print(f"Serial:          {sdr.get_serial()}")
        print(f"Device speed:    {sdr.device_speed}")
        print(f"FPGA size:       {sdr.fpga_size} KLE")
        print(f"FPGA configured: {sdr.fpga_configured}")
        try:
            print(f"RX1 SR range:    {analyzer.rx_channel.sample_rate_range}")
        except Exception:
            pass
        finish(0)

    def run_scan():
        if not do_connect():
            return
        analyzer.start_freq_edit.setText(str(args.start))
        analyzer.stop_freq_edit.setText(str(args.stop))
        analyzer.step_edit.setText(str(args.step))
        analyzer.gain_spin.setValue(args.gain)
        analyzer.samples_combo.setCurrentText(str(args.fft_size))
        analyzer.rx_channel_combo.setCurrentIndex(args.rx_channel - 1)

        analyzer.toggle_live_scanning()
        if not analyzer.live_scanning:
            return fail("Failed to start scanning")

        if getattr(args, "cal_file", None):
            ok, msg = analyzer.load_calibration_from_file(args.cal_file)
            if not ok:
                print(f"Warning: {msg} — continuing without calibration")
            else:
                print(f"Applied calibration: {msg}")

        print(f"Scanning {args.start}–{args.stop} MHz "
              f"(step {args.step} MHz, gain {args.gain} dB) for {args.duration}s...")

        def finalize():
            freqs = analyzer.common_freq
            spectrum = analyzer.composite_spectrum
            if analyzer.live_scanning:
                analyzer.toggle_live_scanning()

            if freqs is None or spectrum is None or len(freqs) == 0:
                return fail("No spectrum data captured")

            with open(args.output, "w") as f:
                f.write("freq_mhz,power_dbm\n")
                for fmhz, p in zip(freqs, spectrum):
                    f.write(f"{fmhz:.4f},{p:.2f}\n")

            peak = int(np.argmax(spectrum))
            print(f"Spectrum saved: {args.output} ({len(freqs)} points)")
            print(f"Peak: {freqs[peak]:.2f} MHz @ {spectrum[peak]:.1f} dBm")
            print(f"Range: {spectrum.min():.1f} .. {spectrum.max():.1f} dBm")
            finish(0)

        QTimer.singleShot(int(args.duration * 1000), finalize)

    def run_iq():
        if not do_connect():
            return
        analyzer.gain_spin.setValue(args.gain)
        # apply gain on the live channel
        try:
            with analyzer.sdr_lock:
                analyzer.rx_channel.gain = args.gain
        except Exception as e:
            print(f"Warning: could not set gain: {e}")

        filepath, fmt = _iq_fmt_from_path(args.output)
        analyzer._iq_begin_recording(filepath, fmt, args.freq * 1e6, args.duration)

        def poll_done():
            if analyzer.iq_recording:
                QTimer.singleShot(50, poll_done)
            else:
                finish(0)
        QTimer.singleShot(100, poll_done)

    def run_tx():
        if not do_connect():
            return
        analyzer.tx_freq_edit.setText(str(args.freq))
        analyzer.tx_gain_spin.setValue(args.gain)
        analyzer.tx_channel_combo.setCurrentIndex(args.tx_channel - 1)

        analyzer.start_transmission()
        if not analyzer.tx_enabled:
            return fail("Failed to start transmission")

        print(f"TX{args.tx_channel}: tone at {args.freq} MHz, "
              f"gain {args.gain} dB, holding {args.duration}s...")

        def stop():
            if analyzer.tx_enabled:
                analyzer.start_transmission()  # toggle off
            print("TX stopped")
            finish(0)
        QTimer.singleShot(int(args.duration * 1000), stop)

    def run_calibrate():
        if not do_connect():
            return

        sr_hz = args.sample_rate * 1e6
        center_hz = args.freq * 1e6

        analyzer.gain_spin.setValue(args.gain)
        analyzer.samples_combo.setCurrentText(str(args.fft_size))

        try:
            with analyzer.sdr_lock:
                analyzer.rx_channel.enable = False
                time.sleep(0.05)
                analyzer.rx_channel.sample_rate = int(sr_hz)
                analyzer.rx_channel.bandwidth = int(sr_hz)
                actual_sr = float(analyzer.rx_channel.sample_rate)
                analyzer.rx_channel.gain = args.gain
                analyzer.rx_channel.frequency = int(center_hz)

                analyzer.config.sample_rate = actual_sr
                analyzer.config.num_samples = args.fft_size
                analyzer.config.gain = args.gain
                analyzer.center_freq = center_hz

                analyzer.sdr.sync_config(
                    layout=_bladerf.ChannelLayout.RX_X1,
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=analyzer.config.SYNC_NUM_BUFFERS,
                    buffer_size=analyzer.config.num_samples *
                                analyzer.config.BUFFER_SIZE_MULTIPLIER,
                    num_transfers=analyzer.config.SYNC_NUM_TRANSFERS,
                    stream_timeout=analyzer.config.SYNC_STREAM_TIMEOUT,
                )
                analyzer.rx_channel.enable = True
                time.sleep(0.1)
        except Exception as e:
            return fail(f"Failed to configure RX for calibration: {e}")

        print(f"Calibrating at {args.freq} MHz "
              f"(actual SR {actual_sr/1e6:.3f} MHz, FFT {args.fft_size}, "
              f"gain {args.gain} dB, {args.averages} averages)…")

        try:
            analyzer.run_calibration(averages=args.averages)
        except Exception as e:
            return fail(f"Calibration failed: {e}")

        if analyzer.segment_correction is None:
            return fail("Calibration produced no result")

        try:
            analyzer.save_calibration_to_file(args.output)
        except Exception as e:
            return fail(f"Save failed: {e}")

        c = analyzer.segment_correction
        print(f"Correction range: {c.min():+.2f} .. {c.max():+.2f} dB "
              f"(std {c.std():.2f} dB)")
        finish(0)

    dispatch = {"info": run_info, "scan": run_scan,
                "iq": run_iq, "tx": run_tx,
                "calibrate": run_calibrate}
    QTimer.singleShot(0, dispatch[args.command])

    app.exec_()
    return exit_code["value"]


# ============================================================================
# Main Entry Point
# ============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="BladeRF 2.0 wideband spectrum analyzer + signal generator. "
                    "Launches the GUI by default; use a subcommand for headless "
                    "automation."
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("info", help="Connect, print device info, exit")

    p_scan = sub.add_parser("scan",
                            help="Run a wideband scan and dump spectrum to CSV")
    p_scan.add_argument("--start", type=float, default=2400,
                        help="Start frequency in MHz (default: 2400)")
    p_scan.add_argument("--stop", type=float, default=2500,
                        help="Stop frequency in MHz (default: 2500)")
    p_scan.add_argument("--step", type=float, default=50,
                        help="Segment step in MHz (default: 50)")
    p_scan.add_argument("--gain", type=int, default=30,
                        help="RX gain in dB, 0–60 (default: 30)")
    p_scan.add_argument("--fft-size", type=int, default=4096,
                        choices=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
                        help="FFT size (default: 4096)")
    p_scan.add_argument("--rx-channel", type=int, default=1, choices=[1, 2],
                        help="RX channel 1 or 2 (default: 1)")
    p_scan.add_argument("--duration", type=float, default=5,
                        help="How long to scan, in seconds (default: 5)")
    p_scan.add_argument("--output", default="/tmp/spectrum.csv",
                        help="Output CSV path (default: /tmp/spectrum.csv)")
    p_scan.add_argument("--cal-file", default=None,
                        help="Path to a .npz calibration file produced by the "
                             "'calibrate' command (or GUI Save Calibration). "
                             "Applies filter-shape correction to the scan.")

    p_cal = sub.add_parser("calibrate",
                           help="Run filter-flattening calibration and save the profile")
    p_cal.add_argument("--freq", type=float, default=2700,
                       help="Center frequency in MHz; pick a quiet band "
                            "(default: 2700)")
    p_cal.add_argument("--sample-rate", type=float, default=30,
                       help="Sample rate in MHz (default: 30)")
    p_cal.add_argument("--fft-size", type=int, default=4096,
                       choices=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
                       help="FFT size (default: 4096)")
    p_cal.add_argument("--gain", type=int, default=40,
                       help="RX gain in dB, 0–60 (default: 40)")
    p_cal.add_argument("--averages", type=int, default=200,
                       help="Number of spectra to average (default: 200)")
    p_cal.add_argument("--output", default="/tmp/bladerf_cal.npz",
                       help="Output .npz path (default: /tmp/bladerf_cal.npz)")

    p_iq = sub.add_parser("iq", help="Record IQ samples to file")
    p_iq.add_argument("--freq", type=float, default=2400,
                      help="Center frequency in MHz (default: 2400)")
    p_iq.add_argument("--duration", type=int, default=200,
                      help="Capture duration in ms (default: 200)")
    p_iq.add_argument("--gain", type=int, default=30,
                      help="RX gain in dB (default: 30)")
    p_iq.add_argument("--output", default="/tmp/iq.npy",
                      help="Output path; extension picks format "
                           "(.npy / .bin / .csv). Default: /tmp/iq.npy")

    p_tx = sub.add_parser("tx", help="Transmit a single 1 kHz-offset tone")
    p_tx.add_argument("--freq", type=float, default=2400,
                      help="TX center frequency in MHz (default: 2400)")
    p_tx.add_argument("--gain", type=int, default=10,
                      help="TX gain in dB (default: 10)")
    p_tx.add_argument("--tx-channel", type=int, default=1, choices=[1, 2],
                      help="TX channel 1 or 2 (default: 1)")
    p_tx.add_argument("--duration", type=float, default=2,
                      help="Transmit time in seconds (default: 2)")

    return p


def main():
    """Main application entry point"""
    args = _build_arg_parser().parse_args()

    QLocale.setDefault(QLocale("C"))

    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.txt")
    log_path = os.path.join(base_path, log_filename)

    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger

    print(f"=================================================================")
    print(f"BladeRF 2.0 Wideband Spectrum Analyzer")
    print(f"Mode: {'headless:' + args.command if args.command else 'GUI'}")
    print(f"Log file: {log_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"=================================================================")

    if args.command is not None:
        sys.exit(run_headless(args))

    # ---- GUI mode (default) ----
    print("Creating QApplication...")
    app = QApplication(sys.argv)
    print("QApplication created")

    if getattr(sys, 'frozen', False):
        icon_path = os.path.join(sys._MEIPASS, "bladerf2_0.ico")
    else:
        icon_path = os.path.join(base_path, "bladerf2_0.ico")

    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        print(f"Icon loaded: {icon_path}")
    else:
        print(f"Warning: Icon not found at {icon_path}")

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

        try:
            QMessageBox.critical(None, "Fatal Error",
                               f"Application failed to start:\n{str(e)}\n\n"
                               f"Check log file for details.")
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()