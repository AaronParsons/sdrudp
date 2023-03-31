#! /usr/bin/env python
import numpy as np
import time
import matplotlib.pyplot as plt
from sdrudp.sdr import SDR
from sdrudp.resample import resample

NSAMPLES = 2048
NBLOCKS = 40
SAMPLE_RATE = 3.2e6
LO = 901e6
GAIN = 0
TONE = 901.5e6

PLOT = True
t = np.arange(NSAMPLES) / SAMPLE_RATE
omega = 2 * np.pi * (TONE - LO)
tone_cos = np.cos(omega * t)
tone_sin = -np.sin(omega * t)  # conjugate


running = True


def on_close(event):
    global running
    running = False


if PLOT:
    plt.ion()
    fig, axs = plt.subplots(nrows=2, sharex=True, constrained_layout=True)
    fig.canvas.mpl_connect("close_event", on_close)
    (line0,) = axs[0].plot(t, np.ones(NSAMPLES))
    (line1,) = axs[0].plot(t, np.ones(NSAMPLES))
    (line2,) = axs[1].plot(t, np.ones(NSAMPLES), c="blue")
    (line3,) = axs[1].plot(t, np.ones(NSAMPLES), c="black")
    axs[0].set_ylim(-128, 128)
    axs[0].set_ylabel("Voltage")
    axs[1].set_ylim(-np.pi, np.pi)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Phase [rad]")
    plt.grid()

sdr = SDR(direct=False, center_freq=LO, sample_rate=SAMPLE_RATE, gain=GAIN)

#t0 = time.time()
while running:
    try:
        data = sdr.capture_data(nsamples=NSAMPLES, nblocks=NBLOCKS)
        #t1 = time.time()
        #print((t1 - t0) / (NSAMPLES * NBLOCKS / SAMPLE_RATE))
        #t0 = t1
        data = data - np.mean(data, axis=1, keepdims=True)  # remove DC offset
        real = data[0, :, 0]
        imag = data[0, :, 1]
        data_cos = real * tone_cos - imag * tone_sin
        data_sin = real * tone_sin + imag * tone_cos
        dphi = np.arctan2(data_sin, data_cos)
        dphi -= dphi[0]  # remove phase offset
        den = np.unwrap(dphi) + omega * t
        nonzero = den != 0
        sample_ADC = SAMPLE_RATE * omega * t[nonzero] / den[nonzero]
        sample_ADC = np.mean(sample_ADC)
        resamp_real = resample(real, sample_ADC, SAMPLE_RATE)
        resamp_imag = resample(imag, sample_ADC, SAMPLE_RATE)
        rs_cos = resamp_real * tone_cos - resamp_imag * tone_sin
        rs_sin = resamp_real * tone_sin + resamp_imag * tone_cos
        rs_dphi = np.arctan2(rs_sin, rs_cos)

        if PLOT:
            line0.set_ydata(data_cos)
            line1.set_ydata(data_sin)
            line2.set_ydata(dphi)
            line3.set_ydata(rs_dphi)
            # dt = np.arctan2(data_sin, data_cos) / (omega / SAMPLE_RATE)
            # line2.set_ydata(dt / np.pi * 128)
            # print(np.mean(np.diff(dt)[:500]))
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        break
