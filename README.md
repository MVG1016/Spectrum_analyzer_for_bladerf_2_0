# Real-Time Wideband Spectrum Analyzer and Signal Generator with BladeRF

This project is a real-time wideband spectrum analyzer and signal generator built using the [BladeRF 2.0](https://www.nuand.com/bladerf-2-0-micro/) software-defined radio and Python.

Before using app you need to install all the necessary drivers and dll files for your BladeRF2.0


## Features

- Real-time spectrum display that allows wideband scan (75-6000MHz)
- Adjustable gain and FFT window settings and frequency range
- Waterfall mode
- "Max Hold" mode for capturing peak values
- Switching between Rx1/Rx2 and Tx1/Tx2 channels
- Highlighting max point on graph
- TX/RX mode switching
- Logging of all console output to a log file
- Tx mode that allows you to transmit at sweep and classical modes

## How to do level calibration aka smoothing
- Start scanning on your desired band
- Hit pause
- Hit calibrate button
- Hit start button again
- You need to recalibrate every time you change any Rx settings
- Calibration is not mandatory but it will remove effects of inner filter in bladeRF and make spectre appear smoother


## To Do list

- Calibrate waterfall so it will readjust colors based on noise level
- Add ability to save IQ samples on separate file
- Split one main.py file into several for convenience

