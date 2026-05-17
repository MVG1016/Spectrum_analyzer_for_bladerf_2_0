"""Headless (CLI) mode for the BladeRF spectrum analyzer.

Reuses SpectrumAnalyzer's device-control methods, driving them through
QTimer callbacks instead of UI events.
"""
import os
import sys
import time

import numpy as np
from bladerf import _bladerf
from PyQt5.QtCore import QLocale, QTimer
from PyQt5.QtWidgets import QApplication

from .abscal import get_calibration, update_absolute_calibration_csv, _abs_cal_default_path
from .analyzer import SpectrumAnalyzer


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

    def run_abscal():
        if not do_connect():
            return

        sr_hz = args.sample_rate * 1e6
        freq_hz = args.freq * 1e6

        try:
            with analyzer.sdr_lock:
                analyzer.rx_channel.enable = False
                time.sleep(0.05)
                analyzer.rx_channel.sample_rate = int(sr_hz)
                analyzer.rx_channel.bandwidth = int(sr_hz)
                actual_sr = float(analyzer.rx_channel.sample_rate)
                analyzer.rx_channel.gain = args.gain
                analyzer.rx_channel.frequency = int(freq_hz)

                analyzer.config.sample_rate = actual_sr
                analyzer.config.gain = args.gain
                analyzer.center_freq = freq_hz

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

                flush_buf = bytearray(analyzer.config.num_samples * 4)
                for _ in range(80):
                    try:
                        analyzer.sdr.sync_rx(flush_buf, analyzer.config.num_samples)
                    except Exception:
                        break
        except Exception as e:
            return fail(f"RX setup failed: {e}")

        n = analyzer.config.num_samples
        print(f"Averaging {args.averages} spectra at {args.freq} MHz, "
              f"SR {actual_sr/1e6:.3f} MHz, gain {args.gain} dB…")

        accum = np.zeros(n, dtype=np.float64)
        for _ in range(args.averages):
            _, p = analyzer.acquire_one_spectrum()
            accum += 10.0 ** (p / 10.0)
        avg = 10.0 * np.log10(accum / args.averages)

        # Find peak, ignoring filter-rolloff edges (10%) and DC LO-leakage bins (±2)
        masked = avg.copy()
        masked[: n // 10] = -200.0
        masked[-n // 10:] = -200.0
        dc = n // 2
        masked[max(0, dc - 2): dc + 3] = -200.0
        peak_idx = int(np.argmax(masked))
        reported = float(avg[peak_idx])

        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / actual_sr))
        peak_freq_mhz = (freqs[peak_idx] + freq_hz) / 1e6

        current_offset = get_calibration(freq_hz)
        new_offset = args.known_power - reported + current_offset
        delta = new_offset - current_offset

        print(f"Peak found at:   {peak_freq_mhz:.4f} MHz")
        print(f"Reported power:  {reported:+.2f} dBm")
        print(f"Known input:     {args.known_power:+.2f} dBm")
        print(f"Current offset:  {current_offset:+.2f} dB @ {args.freq} MHz")
        print(f"New offset:      {new_offset:+.2f} dB  (Δ {delta:+.2f} dB)")

        if not args.dry_run:
            try:
                update_absolute_calibration_csv(freq_hz, new_offset)
                print(f"Wrote {_abs_cal_default_path()}")
            except Exception as e:
                return fail(f"Could not update CSV: {e}")
        else:
            print("(--dry-run set, CSV not modified)")
        finish(0)

    dispatch = {"info": run_info, "scan": run_scan,
                "iq": run_iq, "tx": run_tx,
                "calibrate": run_calibrate,
                "abscal": run_abscal}
    QTimer.singleShot(0, dispatch[args.command])

    app.exec_()
    return exit_code["value"]


