from bladerf import _bladerf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator

# --- Параметры сканирования ---  # --- Scanning Parameters ---
START_FREQ = 70e6   # Начальная частота  # Start frequency
END_FREQ = 6e9    # Конечная частота  # End frequency
BANDWIDTH = 40e6    # Ширина полосы  # Bandwidth
STEP = 40e6         # Шаг сканирования  # Step size
SAMPLE_RATE = 40e6  # Частота дискретизации  # Sample rate
NUM_SAMPLES = 8192  # Количество сэмплов  # Number of samples
GAIN = 0            # Усиление (в dB)  # Gain (in dB)
WATERFALL_SIZE = 100  # Количество строк в водопаде  # Number of lines in waterfall

# --- Инициализация BladeRF ---  # --- BladeRF Initialization ---
sdr = _bladerf.BladeRF()
rx_ch = sdr.Channel(_bladerf.CHANNEL_RX(0))

# Проверка допустимого диапазона частот  # Check valid frequency range
freq_range = rx_ch.frequency_range
min_freq = int(freq_range.min)
max_freq = int(freq_range.max)

#print(f"Min Frequency: {min_freq / 1e6} MHz")  # Uncomment if needed for debugging
#print(f"Max Frequency: {max_freq / 1e6} MHz")  # Uncomment if needed for debugging

# --- Настройка RX ---  # --- RX Configuration ---
rx_ch.sample_rate = int(SAMPLE_RATE)
rx_ch.bandwidth = int(BANDWIDTH)
rx_ch.gain_mode = _bladerf.GainMode.Manual
rx_ch.gain = GAIN

# --- Генерация списка частот ---  # --- Frequency List Generation ---
frequencies = np.array([f for f in np.arange(START_FREQ, END_FREQ, STEP) if min_freq <= f <= max_freq])
spectrum = np.zeros_like(frequencies, dtype=float)

print(f"Scanning {len(frequencies)} frequency steps...")

# --- Настройка потока ---  # --- Stream Configuration ---
buffer_size = NUM_SAMPLES * 4
sdr.sync_config(
    layout=_bladerf.ChannelLayout.RX_X1,
    fmt=_bladerf.Format.SC16_Q11,
    num_buffers=16,
    buffer_size=buffer_size,
    num_transfers=8,
    stream_timeout=3500
)

# Включение RX  # Enable RX
rx_ch.enable = True

# --- Настройка графиков ---  # --- Plot Configuration ---
fig = plt.figure(figsize=(12, 8))
# Используем GridSpec для управления расположением графиков  # Using GridSpec for plot layout
gs = plt.GridSpec(3, 1, height_ratios=[2, 0.1, 1])  # 2 parts for spectrum, 0.1 for colorbar, 1 for waterfall

# График спектра  # Spectrum Plot
ax1 = fig.add_subplot(gs[0])
line, = ax1.plot(frequencies / 1e6, spectrum, color='b', linewidth=1)
ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("Power (dBm)")
ax1.set_title("BladeRF 2.0 Live Spectrum Scan")
ax1.grid(True)

# Инициализация маркера для максимального значения  # Initialize marker for max value
max_marker, = ax1.plot([], [], 'ro', label='Max Power')  # Красный маркер для максимума  # Red marker for max power
max_text = ax1.text(0.05, 0.95, "", transform=ax1.transAxes, ha="left", va="top", fontsize=12, color='r')

# Водопад  # Waterfall
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

# Цветовая шкала (colorbar) для водопада  # Colorbar for waterfall
cbar = plt.colorbar(waterfall, cax=fig.add_subplot(gs[1]), orientation='horizontal', label="Power (dBm)")

# --- Функция обновления графика с авто-масштабированием ---  # --- Update Function with Auto-scaling ---
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

    # Обновляем данные графика спектра  # Update spectrum plot data
    line.set_ydata(spectrum)

    # Находим индекс максимального значения в спектре  # Find the index of the max value in spectrum
    max_index = np.argmax(spectrum)
    max_freq_value = frequencies[max_index] / 1e6  # Частота максимума в MHz  # Frequency of maximum in MHz
    max_power_value = spectrum[max_index]  # Мощность на максимуме  # Power at maximum

    # Обновляем маркер  # Update marker
    max_marker.set_data(max_freq_value, max_power_value)

    # Обновляем текст с координатами маркера  # Update marker coordinate text
    max_text.set_text(f"Max: {max_freq_value:.2f} MHz, {max_power_value:.2f} dBm")

    # Автоматическое обновление пределов осей  # Auto-update axis limits
    ax1.relim()  # Пересчитывает границы осей  # Recalculate axis limits
    ax1.autoscale_view()  # Обновляет пределы  # Update axis view

    # Устанавливаем пределы оси X точно в диапазоне частот  # Set X-axis limits exactly in frequency range
    ax1.set_xlim(frequencies[0] / 1e6, frequencies[-1] / 1e6)

    # Используем MaxNLocator для автоматической установки меток оси X  # Use MaxNLocator for automatic X-axis tick placement
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Устанавливаем дополнительные метки для первой и последней частоты  # Set additional ticks for the first and last frequency
    ax1.set_xticks([frequencies[0] / 1e6, frequencies[-1] / 1e6] + ax1.get_xticks().tolist())

    # Обновляем водопад  # Update waterfall
    waterfall_data = np.roll(waterfall_data, -1, axis=0)  # Сдвигаем данные вверх  # Shift data up
    waterfall_data[-1, :] = spectrum  # Добавляем новую строку вниз  # Add new row at the bottom
    waterfall.set_data(waterfall_data)
    waterfall.set_clim(vmin=np.min(spectrum), vmax=np.max(spectrum))  # Обновляем диапазон цветов  # Update color range

    return line, max_marker, max_text, waterfall  # Возвращаем маркер, линию и водопад для обновления  # Return marker, line, and waterfall for update

# --- Анимация обновления графика ---  # --- Animation for updating plot ---
ani = animation.FuncAnimation(fig, update, interval=500, blit=False)

plt.tight_layout()
plt.show()

# Завершение работы  # End work
rx_ch.enable = False
sdr.close()
