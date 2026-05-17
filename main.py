"""
BladeRF 2.0 Spectrum Analyzer entry point.

The actual implementation lives in the bladerf_spec/ package — split out
to keep each module focused (GUI, SDR control, DSP, headless CLI, etc).
"""
from bladerf_spec.cli import main


if __name__ == "__main__":
    main()
