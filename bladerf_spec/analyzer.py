"""The Qt main window: SpectrumAnalyzer.

Owns the BladeRF device, the RX/TX channels, the scan engine, the DSP
pipeline, the calibration profile, the IQ recorder, and the GUI widgets.
"""
import os
import time
from datetime import datetime
from threading import Lock

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QRectF, Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QComboBox, QFormLayout,
    QLineEdit, QLabel, QMessageBox, QSpinBox, QFrame,
    QSlider, QGroupBox, QFileDialog, QScrollArea,
)

from bladerf import _bladerf

from .abscal import get_calibration
from .config import SDRConfig
from .tx_thread import TXThread


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

        # IQ recording — chunks streamed straight to disk, no accumulator
        self.iq_recording = False
        self.iq_save_path = None
        self.iq_total_samples = 0
        self._iq_file = None

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
                    except Exception as e:
                        print(f"Warning: RX channel disable failed: {e}")
                    self.rx_channel = None

                if self.tx_channel is not None:
                    try:
                        self.tx_channel.enable = False
                    except Exception as e:
                        print(f"Warning: TX channel disable failed: {e}")
                    self.tx_channel = None

                if self.sdr is not None:
                    try:
                        self.sdr.close()
                    except Exception as e:
                        print(f"Warning: SDR close failed: {e}")
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
        if not self.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect to BladeRF first")
            return
        if self.sdr is None:
            QMessageBox.warning(self, "Error", "BladeRF not initialized")
            return
        if self.live_scanning:
            QMessageBox.warning(self, "Error", "Stop scanning before calibration")
            return
        if self.wb_centers is None or len(self.wb_centers) == 0:
            QMessageBox.warning(self, "Error",
                                "Configure scan range first and press Start/Pause before calibrating")
            return

        self.calibrating = True
        self.calibrate_button.setEnabled(False)
        self.segment_correction = None
        self.segment_correction_map = {}
        QApplication.processEvents()

        CAL_AVERAGES = int(averages)
        total_centers = len(self.wb_centers)

        try:
            for ci, center_hz in enumerate(self.wb_centers):
                self.calibrate_status_label.setText(
                    f"Calibrating {ci + 1}/{total_centers}  {center_hz / 1e6:.1f} MHz...")
                QApplication.processEvents()

                with self.sdr_lock:
                    self.rx_channel.frequency = int(center_hz)
                    self.center_freq = center_hz
                    flush_buf = bytearray(self.config.num_samples * 4)
                    flush_count = (self.config.SYNC_NUM_BUFFERS *
                                   self.config.BUFFER_SIZE_MULTIPLIER + 4)
                    for _ in range(flush_count):
                        try:
                            self.sdr.sync_rx(flush_buf, self.config.num_samples)
                        except Exception:
                            break

                accum = np.zeros(self.config.num_samples, dtype=np.float64)

                for i in range(CAL_AVERAGES):
                    _, power_dbm = self.acquire_one_spectrum()

                    # Детальная диагностика первого кадра каждого сегмента
                    if i == 0:
                        nan_count = np.sum(~np.isfinite(power_dbm))
                        print(f"  Seg {ci + 1} frame 0: shape={power_dbm.shape} "
                              f"dtype={power_dbm.dtype} "
                              f"min={np.nanmin(power_dbm):.1f} "
                              f"max={np.nanmax(power_dbm):.1f} "
                              f"nan={nan_count}")

                    power_dbm = np.where(np.isfinite(power_dbm), power_dbm, -140.0)
                    accum += power_dbm

                    if i % 20 == 0:
                        self.calibrate_status_label.setText(
                            f"Calibrating {ci + 1}/{total_centers}  "
                            f"{center_hz / 1e6:.1f} MHz  {i}/{CAL_AVERAGES}")
                        QApplication.processEvents()

                avg_profile = accum / CAL_AVERAGES
                print(f"  Seg {ci + 1} avg_profile: "
                      f"min={np.nanmin(avg_profile):.1f} "
                      f"max={np.nanmax(avg_profile):.1f} "
                      f"nan={np.sum(~np.isfinite(avg_profile))}")

                avg_profile = np.where(np.isfinite(avg_profile), avg_profile, -140.0)

                n = len(avg_profile)
                ref_level = np.median(avg_profile[n // 3: 2 * n // 3])
                print(f"  Seg {ci + 1} ref_level={ref_level:.1f}")

                correction = avg_profile - ref_level
                self.segment_correction_map[center_hz] = correction

            self.segment_correction = list(self.segment_correction_map.values())[-1]
            self.calibrate_status_label.setText(f"OK  {total_centers} segments calibrated")
            print(f"Calibration done: {total_centers} segments")

        except Exception as e:
            print(f"Calibration error: {e}")
            import traceback
            traceback.print_exc()
            self.segment_correction = None
            self.segment_correction_map = {}
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
                finite_mask = np.isfinite(self.composite_spectrum)
                if finite_mask.any():
                    masked = np.where(finite_mask, self.composite_spectrum, -np.inf)
                    idx = np.argmax(masked)
                    f = self.common_freq[idx]
                    p = self.composite_spectrum[idx]
                    if np.isfinite(f) and np.isfinite(p):
                        self.max_marker.setData([f], [p])
                        self.max_text.setText(f"{f:.2f} MHz\n{p:.1f} dBm")
                        self.max_text.setPos(f, p)
                else:
                    self.max_marker.setData([], [])
                    self.max_text.setText("")

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
                self.tx_thread.error.connect(self._on_tx_thread_error)
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
                    self.tx_thread.error.connect(self._on_tx_thread_error)
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

    def _on_tx_thread_error(self, message: str, error_count: int):
        """Reflect TX-thread errors into the GUI status label."""
        try:
            self.tx_status_label.setText(
                f"TX error #{error_count}: {message[:60]}"
            )
        except Exception:
            pass  # widget may not exist in some test paths

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

        Streams each chunk straight to disk — memory stays constant
        regardless of recording duration.
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
        self.iq_total_samples = 0

        try:
            self._iq_open_stream(filepath, fmt, total_samples)
        except Exception as e:
            print(f"IQ open error: {e}")
            self.iq_status_label.setText(f"Open error: {e}")
            self.iq_record_button.setEnabled(True)
            return

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

    def _iq_open_stream(self, filepath: str, fmt: str, total_samples: int):
        """Open output file, write headers if needed."""
        if fmt == "npy":
            # Pre-write the .npy header for the known total length so the
            # body can just be raw little-endian complex64 bytes appended.
            from numpy.lib.format import write_array_header_2_0, dtype_to_descr
            self._iq_file = open(filepath, "wb")
            header = {
                "descr": dtype_to_descr(np.dtype("<c8")),
                "fortran_order": False,
                "shape": (total_samples,),
            }
            write_array_header_2_0(self._iq_file, header)
        elif fmt == "csv":
            self._iq_file = open(filepath, "w", buffering=1 << 16)
            self._iq_file.write("real,imag\n")
        else:  # bin
            self._iq_file = open(filepath, "wb")

    def _iq_write_chunk(self, samples):
        """Stream one chunk of complex64 samples to the open file."""
        if self.iq_fmt == "npy":
            samples.astype("<c8").tofile(self._iq_file)
        elif self.iq_fmt == "csv":
            for s in samples:
                self._iq_file.write(f"{s.real},{s.imag}\n")
        else:  # bin — int16 interleaved I,Q,I,Q,...
            iq_int16 = np.empty(len(samples) * 2, dtype=np.int16)
            iq_int16[0::2] = np.clip(samples.real * 2047, -2048, 2047).astype(np.int16)
            iq_int16[1::2] = np.clip(samples.imag * 2047, -2048, 2047).astype(np.int16)
            iq_int16.tofile(self._iq_file)

    def _iq_close_stream(self):
        f = getattr(self, "_iq_file", None)
        if f is not None:
            try:
                f.close()
            except Exception as e:
                print(f"Warning: IQ file close failed: {e}")
            self._iq_file = None

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
            self._iq_write_chunk(samples)
            self.iq_reads_done += 1
            self.iq_total_samples += len(samples)

            progress = int(self.iq_reads_done / self.iq_reads_needed * 100)
            self.iq_status_label.setText(f"Recording... {progress}%")

            if self.iq_reads_done < self.iq_reads_needed:
                QTimer.singleShot(0, self._iq_read_chunk)
            else:
                self._iq_finalize()

        except Exception as e:
            print(f"IQ read error: {e}")
            import traceback
            traceback.print_exc()
            self._iq_close_stream()
            self.iq_recording = False
            self.iq_record_button.setEnabled(True)
            self.iq_status_label.setText(f"Error: {e}")

    def _iq_finalize(self):
        """Close the stream, write metadata, restore scanning if needed."""
        try:
            self._iq_close_stream()

            filepath = self.iq_save_path
            fmt = self.iq_fmt
            n = self.iq_total_samples

            meta_path = filepath.rsplit(".", 1)[0] + "_meta.txt"
            with open(meta_path, "w") as f:
                f.write(f"center_freq_hz={self.iq_center_hz}\n")
                f.write(f"sample_rate_hz={self.config.sample_rate}\n")
                f.write(f"num_samples={n}\n")
                f.write(f"format={fmt}\n")
                f.write(f"gain={self.config.gain}\n")
                f.write(f"recorded_at={datetime.now().isoformat()}\n")

            size_kb = os.path.getsize(filepath) / 1024
            print(f"IQ saved: {filepath}  ({n} samples, {size_kb:.1f} KB)")
            print(f"Metadata: {meta_path}")

            self.iq_status_label.setText(
                f"Saved {n} samples  ({size_kb:.0f} KB)"
            )

        except Exception as e:
            print(f"IQ finalize error: {e}")
            import traceback
            traceback.print_exc()
            self.iq_status_label.setText(f"Save error: {e}")

        finally:
            self.iq_recording = False
            self.iq_record_button.setEnabled(True)

            if self._iq_was_scanning:
                QTimer.singleShot(200, lambda: self.scan_button.click())

    def closeEvent(self, event):
        """Handle application close"""
        print("Closing application...")

        if self.is_connected:
            self.disconnect_bladerf()

        print("Application closed successfully")
        event.accept()


