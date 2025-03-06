# Live Spectrum Scan with BladeRF

This project demonstrates how to use the BladeRF 2.0 device to perform a live spectrum scan and visualize the results in real-time using Python. It includes an animated plot that updates continuously with the received data, highlighting the maximum power point on the spectrum.
Features

    Live spectrum scanning using the BladeRF 2.0 device.
    Real-time plotting with matplotlib and live updates.
    Automatic scaling of the plot based on the received data.
    Marker indicating the maximum power point in the spectrum.
    Display of frequency and power values of the maximum point in the legend.

Requirements

    Python 3.6+
    BladeRF 2.0 device
    bladerf Python library
    numpy
    matplotlib

Install the required libraries using pip:

>pip install bladerf numpy matplotlib

Code Explanation
Parameters:

    START_FREQ: The starting frequency for the scan (70 MHz).
    END_FREQ: The end frequency for the scan (6 GHz).
    BANDWIDTH: The bandwidth for the scan (40 MHz).
    STEP: The frequency step size (40 MHz).
    SAMPLE_RATE: The sample rate (40 MHz).
    NUM_SAMPLES: Number of samples per scan.
    GAIN: Gain setting in dB.

Flow:

    BladeRF Initialization: The BladeRF 2.0 device is initialized and the RX channel is configured.
    Frequency Range: The script checks the valid frequency range of the device.
    Real-Time Scanning: The script loops through the frequency range, captures samples, and computes the power in dBm for each frequency.
    Plotting: The data is plotted using matplotlib, and the plot updates in real-time with an animation. The maximum power point is marked on the plot, and the frequency and power values of this point are displayed in the legend.

Usage

    Connect your BladeRF 2.0 device to your computer.
    Run the Python script.

python spectrum_scan.py

    A live plot will appear, showing the spectrum and updating in real-time with the received data.

Russian
# Сканирование спектра с BladeRF

Этот проект демонстрирует, как использовать устройство BladeRF 2.0 для выполнения сканирования спектра и визуализации результатов в реальном времени с использованием Python. Он включает анимированный график, который обновляется непрерывно с полученными данными, выделяя точку максимальной мощности на спектре.
Особенности

    Живое сканирование спектра с использованием устройства BladeRF 2.0.
    Реальное время обновления графика с использованием matplotlib.
    Автоматическая настройка масштаба графика на основе полученных данных.
    Маркер, указывающий на точку максимальной мощности на спектре.
    Отображение частоты и значения мощности точки максимума в легенде.

Требования

    Python 3.6+
    Устройство BladeRF 2.0
    Библиотека bladerf для Python
    numpy
    matplotlib

Установите необходимые библиотеки с помощью pip:

>pip install bladerf numpy matplotlib

Объяснение кода
Параметры:

    START_FREQ: Начальная частота для сканирования (70 МГц).
    END_FREQ: Конечная частота для сканирования (6 ГГц).
    BANDWIDTH: Ширина полосы для сканирования (40 МГц).
    STEP: Шаг частоты (40 МГц).
    SAMPLE_RATE: Частота дискретизации (40 МГц).
    NUM_SAMPLES: Количество сэмплов на сканирование.
    GAIN: Усиление в dB.

Порядок выполнения:

    Инициализация BladeRF: Устройство BladeRF 2.0 и RX канал конфигурируются.
    Диапазон частот: Скрипт проверяет допустимый диапазон частот устройства.
    Живое сканирование: Скрипт выполняет сканирование через диапазон частот, захватывает сэмплы и вычисляет мощность в dBm для каждой частоты.
    Построение графика: Данные отображаются на графике с использованием matplotlib, и график обновляется в реальном времени с анимацией. На графике выделяется точка максимальной мощности, а частота и мощность этой точки отображаются в легенде.

Использование

    Подключите устройство BladeRF 2.0 к компьютеру.
    Запустите Python-скрипт.

python spectrum_scan.py

    Появится живой график, показывающий спектр, который будет обновляться в реальном времени с полученными данными.
