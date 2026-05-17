"""Runtime configuration dataclass."""
from dataclasses import dataclass


@dataclass
class SDRConfig:
    """SDR configuration parameters"""
    start_freq: float = 70e6      # 70 MHz
    stop_freq: float = 6000e6     # 6000 MHz
    step: float = 0.6e6           # 0.53 MHz (equal to sample rate)

    sample_rate: float = step
    num_samples: int = 4096
    gain: int = 30

    # Waterfall
    waterfall_lines: int = 30

    # BladeRF streaming
    SYNC_NUM_BUFFERS: int = 16
    SYNC_NUM_TRANSFERS: int = 8
    SYNC_STREAM_TIMEOUT: int = 3500
    BUFFER_SIZE_MULTIPLIER: int = 4
