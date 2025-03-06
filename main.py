from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Параметры сканирования ---
START_FREQ = 70e6   # Начальная частота (70 МГц)
END_FREQ = 6e9    # Конечная частота (6000 МГц)
BANDWIDTH = 40e6     # Ширина полосы (40 МГц)
STEP = 40e6          # Шаг сканирования (40 МГц)
SAMPLE_RATE = 40e6   # Частота дискретизации (40 МГц)
NUM_SAMPLES = 8192 # Количество сэмплов
GAIN = 0            # Усиление (в dB)

# --- Инициализация BladeRF ---
sdr = _bladerf.BladeRF()
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))

# Проверка допустимого диапазона частот
freq_range = rx_ch.frequency_range
min_freq = int(freq_range.min)
max_freq = int(freq_range.max)

print(f"Min Frequency: {min_freq / 1e6} MHz")
print(f"Max Frequency: {max_freq / 1e6} MHz")

# --- Настройка RX ---
rx_ch.sample_rate = int(SAMPLE_RATE)
rx_ch.bandwidth = int(BANDWIDTH)
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = GAIN

# --- Генерация списка частот ---
frequencies = np.array([f for f in np.arange(START_FREQ, END_FREQ, STEP) if min_freq <= f <= max_freq])
spectrum = np.zeros_like(frequencies, dtype=float)

print(f"Scanning {len(frequencies)} frequency steps...")

# --- Настройка потока ---
buffer_size = NUM_SAMPLES * 4
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

# --- Настройка графика ---
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot(frequencies / 1e6, spectrum, color='b', linewidth=1)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Power (dBm)")
ax.set_title("BladeRF 2.0 Live Spectrum Scan")
ax.grid(True)

# --- Инициализация маркера для максимального значения ---
max_marker, = ax.plot([], [], 'ro', label='Max Power')  # Красный маркер для максимума
max_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=12, color='r')

# --- Функция обновления графика с авто-масштабированием ---
def update(frame):
    global spectrum
    for i, freq in enumerate(frequencies):
        rx_ch.frequency = int(freq)
        buf = bytearray(buffer_size)
        sdr.sync_rx(buf, NUM_SAMPLES)

        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples = samples.view(np.complex64) / (2**11)

        power_mW = np.mean(np.abs(samples)**2)
        spectrum[i] = 10 * np.log10(power_mW)

    # Обновляем данные графика
    line.set_ydata(spectrum)

    # Находим индекс максимального значения в спектре
    max_index = np.argmax(spectrum)
    max_freq_value = frequencies[max_index] / 1e6  # Частота максимума в MHz
    max_power_value = spectrum[max_index]  # Мощность на максимуме

    # Обновляем маркер
    max_marker.set_data(max_freq_value, max_power_value)

    # Обновляем текст с координатами маркера
    max_text.set_text(f"Max: {max_freq_value:.2f} MHz, {max_power_value:.2f} dBm")

    # Автоматическое обновление пределов осей
    ax.relim()  # Пересчитывает границы осей
    ax.autoscale_view()  # Обновляет пределы

    return line, max_marker, max_text  # Возвращаем маркер и линию для обновления

# --- Анимация обновления графика ---
ani = animation.FuncAnimation(fig, update, interval=500, blit=False)

plt.show()

# Завершение работы
rx_ch.enable = False
sdr.close()
