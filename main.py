from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator

# --- Параметры сканирования ---
START_FREQ = 70e6   # Начальная частота
END_FREQ = 6e9      # Конечная частота
BANDWIDTH = 40e6    # Ширина полосы
STEP = 40e6         # Шаг сканирования
SAMPLE_RATE = 40e6  # Частота дискретизации
NUM_SAMPLES = 8192  # Количество сэмплов
GAIN = 0            # Усиление (в dB)
WATERFALL_SIZE = 100  # Количество строк в водопаде

# --- Инициализация BladeRF ---
sdr = _bladerf.BladeRF()
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))

# Проверка допустимого диапазона частот
freq_range = rx_ch.frequency_range
min_freq = int(freq_range.min)
max_freq = int(freq_range.max)

#print(f"Min Frequency: {min_freq / 1e6} MHz")
#print(f"Max Frequency: {max_freq / 1e6} MHz")

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

# --- Настройка графиков ---
fig = plt.figure(figsize=(12, 8))
# Используем GridSpec для управления расположением графиков
gs = plt.GridSpec(3, 1, height_ratios=[2, 0.1, 1])  # 2 части для спектра, 0.1 для colorbar, 1 для водопада

# График спектра
ax1 = fig.add_subplot(gs[0])
line, = ax1.plot(frequencies / 1e6, spectrum, color='b', linewidth=1)
ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("Power (dBm)")
ax1.set_title("BladeRF 2.0 Live Spectrum Scan")
ax1.grid(True)

# Инициализация маркера для максимального значения
max_marker, = ax1.plot([], [], 'ro', label='Max Power')  # Красный маркер для максимума
max_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, ha="left", va="top", fontsize=12, color='r')

# Водопад
ax2 = fig.add_subplot(gs[2])
waterfall_data = np.zeros((WATERFALL_SIZE, len(frequencies)))
waterfall = ax2.imshow(
    waterfall_data,
    aspect='auto',
    cmap='viridis',
    extent=[frequencies[0] / 1e6, frequencies[-1] / 1e6, 0, WATERFALL_SIZE]
)
ax2.set_xlabel("Frequency (MHz)")
ax2.set_ylabel("Time")
ax2.set_title("Waterfall Plot")

# Цветовая шкала (colorbar) для водопада
cbar = plt.colorbar(waterfall, cax=fig.add_subplot(gs[1]), orientation='horizontal', label="Power (dBm)")

# --- Функция обновления графика с авто-масштабированием ---
# --- Функция обновления графика с авто-масштабированием ---
def update(frame):
    global spectrum, waterfall_data

    for i, freq in enumerate(frequencies):
        rx_ch.frequency = int(freq)
        buf = bytearray(buffer_size)
        sdr.sync_rx(buf, NUM_SAMPLES)

        samples = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
        samples = samples.view(np.complex64) / (2**11)

        power_mW = np.mean(np.abs(samples)**2)
        spectrum[i] = 10 * np.log10(power_mW)

    # Обновляем данные графика спектра
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
    ax1.relim()  # Пересчитывает границы осей
    ax1.autoscale_view()  # Обновляет пределы

    # Устанавливаем пределы оси X точно в диапазоне частот
    ax1.set_xlim(frequencies[0] / 1e6, frequencies[-1] / 1e6)

    # Используем MaxNLocator для автоматической установки меток оси X
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Устанавливаем дополнительные метки для первой и последней частоты
    ax1.set_xticks([frequencies[0] / 1e6, frequencies[-1] / 1e6] + ax1.get_xticks().tolist())

    # Обновляем водопад
    waterfall_data = np.roll(waterfall_data, -1, axis=0)  # Сдвигаем данные вверх
    waterfall_data[-1, :] = spectrum  # Добавляем новую строку вниз
    waterfall.set_data(waterfall_data)
    waterfall.set_clim(vmin=np.min(spectrum), vmax=np.max(spectrum))  # Обновляем диапазон цветов

    return line, max_marker, max_text, waterfall  # Возвращаем маркер, линию и водопад для обновления
# --- Анимация обновления графика ---
ani = animation.FuncAnimation(fig, update, interval=500, blit=False)

plt.tight_layout()
plt.show()

# Завершение работы
rx_ch.enable = False
sdr.close()