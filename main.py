from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt

# --- Параметры сканирования ---
START_FREQ = 80e6  # Начальная частота (70 МГц)
END_FREQ = 100e6      # Конечная частота (6 ГГц)
BANDWIDTH = 1e5    # Ширина полосы (40 МГц)
STEP = 1e5         # Шаг сканирования (40 МГц)
SAMPLE_RATE = 7e5  # Частота дискретизации (40 МГц)
NUM_SAMPLES = 32768 # Количество сэмплов на шаге
GAIN = 0           # Усиление в dB

# --- Инициализация BladeRF ---
sdr = _bladerf.BladeRF()
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))  # RX канал 0

# Проверка диапазона допустимых частот
# Получаем диапазон частот
freq_range = rx_ch.frequency_range
print(f"Raw frequency range: {freq_range}")

# Попытка извлечь min и max частоту
try:
    min_freq = int(freq_range.min)
    max_freq = int(freq_range.max)
except AttributeError:
    print("Error: frequency_range does not have min/max attributes!")
    exit(1)

print(f"Min Frequency: {min_freq / 1e6} MHz")
print(f"Max Frequency: {max_freq / 1e6} MHz")


# --- Настройка RX ---
rx_ch.sample_rate = int(SAMPLE_RATE)
rx_ch.bandwidth = int(BANDWIDTH)
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = GAIN

# --- Генерация списка частот (с учетом допустимого диапазона) ---
frequencies = np.array([f for f in np.arange(START_FREQ, END_FREQ, STEP) if min_freq <= f <= max_freq])
spectrum = np.zeros_like(frequencies, dtype=float)
print(f"Scanning {len(frequencies)} frequency steps...")

# --- Настройка синхронного потока ---
buffer_size = NUM_SAMPLES * 4  # Размер буфера под количество сэмплов
sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=buffer_size,
    num_transfers=8,
    stream_timeout=3500
)

# Включение RX
rx_ch.enable = True

# --- Основной цикл сканирования ---
for i, freq in enumerate(frequencies):
    rx_ch.frequency = round(freq / 1e6) * 1e6  # Округление до 1 МГц
    buf = bytearray(buffer_size)
    sdr.sync_rx(buf, NUM_SAMPLES)

    # Преобразование буфера в массив комплексных чисел
    samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    samples = samples.view(np.complex64) / (2**11)  # SC16_Q11 нормализация

    # Вычисление мощности в dBm
    power_mW = np.mean(np.abs(samples)**2) * 1  # Приведение к мВт
    avg_power = np.mean([np.mean(np.abs(samples) ** 2) for _ in range(5)])  # Усредняем 5 замеров
    #spectrum[i] = 10 * np.log10(avg_power)

    spectrum[i] = 20 * np.log10(power_mW)  # Преобразование в dBm

# Выключение RX и закрытие устройства
rx_ch.enable = False
sdr.close()

# --- Визуализация спектра ---
plt.figure(figsize=(14, 6))
plt.plot(frequencies / 1e9, spectrum, label="Spectrum", color='b', linewidth=1)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (dBm)")
plt.title("BladeRF 2.0 Spectrum Scan (70 MHz - 6 GHz)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.show()
