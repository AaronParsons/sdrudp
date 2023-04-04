#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sdrudp.sdr import SDR
from sdrudp.resample import resample

NSAMPLES = 2048
NBLOCKS = 1
SAMPLE_RATE = 2.2e6
LO = 901e6
GAIN = 0
TONE = 901.5e6

PLOT = True
TIMING = False

t = np.arange(NSAMPLES) / SAMPLE_RATE
omega = 2 * np.pi * (TONE - LO)
tone_cos = np.cos(omega * t)
tone_sin = -np.sin(omega * t)  # conjugate

def phase(real, imag, tone_cos, tone_sin):
    data_cos = real * tone_cos - imag * tone_sin
    data_sin = real * tone_sin + imag * tone_cos
    dphi = -np.arctan2(data_sin, data_cos)
    return dphi

running = True


def on_close(event):
    global running
    running = False


if PLOT:
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_close)
    (line0,) = ax.plot(t, np.ones(NSAMPLES), label="original")
    (line1,) = ax.plot(t, np.ones(NSAMPLES), label="resampled")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phase [rad]")
    ax.legend()
    plt.grid()

sdr = SDR(direct=False, center_freq=LO, sample_rate=SAMPLE_RATE, gain=GAIN)

if TIMING:
    import time

    t0 = time.time()

all_data = []
all_rs_data = []
all_dphi = []
all_rs_dphi = []
all_sample_adc = []

while running:
    try:
        data = sdr.capture_data(nsamples=NSAMPLES, nblocks=NBLOCKS)
        if TIMING:
            t1 = time.time()
            print((t1 - t0) / (NSAMPLES * NBLOCKS / SAMPLE_RATE))
            t0 = t1
            continue
        data = data - np.mean(data, axis=1, keepdims=True)  # remove DC offset
        real = data[0, :, 0]
        imag = data[0, :, 1]
        dphi = phase(real, imag, tone_cos, tone_sin)
        dphi -= dphi[0]  # remove phase offset
        den = omega * t - np.unwrap(dphi)
        nonzero = den != 0
        sample_adc = np.mean(SAMPLE_RATE * omega * t[nonzero] / den[nonzero])
        print(sample_adc / 1e6)
        rs_real = resample(real, sample_adc, SAMPLE_RATE)
        rs_imag = resample(imag, sample_adc, SAMPLE_RATE)
        rs_dphi = phase(rs_real, rs_imag, tone_cos, tone_sin)

        all_data.append(data)
        all_rs_data.append(np.array([rs_real, rs_imag]).T)
        all_dphi.append(dphi)
        all_rs_dphi.append(rs_dphi)
        all_sample_adc.append(sample_adc)

        if PLOT:
            line0.set_ydata(dphi)
            line1.set_ydata(rs_dphi)
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        break

sdr.close()
d = {
    "nsamples": NSAMPLES,
    "nblocks": NBLOCKS,
    "sample_rate": SAMPLE_RATE,
    "lo": LO,
    "gain": GAIN,
    "tone": TONE,
    "data": all_data,
    "rs_data": all_rs_data,
    "dphi": all_dphi,
    "rs_dphi": all_rs_dphi,
    "sample_adc": all_sample_adc,
}
np.savez("data.npz", **d)
