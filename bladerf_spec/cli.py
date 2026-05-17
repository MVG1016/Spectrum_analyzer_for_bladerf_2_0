"""CLI entry point: parses args, sets up logging, dispatches to GUI or headless."""
import argparse
import os
import sys
from datetime import datetime

from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMessageBox

from .analyzer import SpectrumAnalyzer
from .headless import run_headless
from .logging import Logger


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

    p_abs = sub.add_parser(
        "abscal",
        help="Measure absolute-power offset against a known reference signal "
             "and write it into absolute_calibration.csv")
    p_abs.add_argument("--freq", type=float, required=True,
                       help="Frequency of the injected reference signal in MHz")
    p_abs.add_argument("--known-power", type=float, required=True,
                       help="True power of the reference at the BladeRF "
                            "antenna port, in dBm (account for cable loss)")
    p_abs.add_argument("--gain", type=int, default=30,
                       help="RX gain in dB (default: 30)")
    p_abs.add_argument("--sample-rate", type=float, default=30,
                       help="Sample rate in MHz (default: 30)")
    p_abs.add_argument("--averages", type=int, default=50,
                       help="Spectra to average (default: 50)")
    p_abs.add_argument("--dry-run", action="store_true",
                       help="Print the computed offset but do not write CSV")

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


def _project_root() -> str:
    """Project root (where main.py and absolute_calibration.csv live)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main application entry point"""
    args = _build_arg_parser().parse_args()

    QLocale.setDefault(QLocale("C"))

    base_path = _project_root()

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

    if getattr(sys, "frozen", False):
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
        except Exception as msgbox_err:
            print(f"(could not show fatal-error dialog: {msgbox_err})")

        sys.exit(1)
