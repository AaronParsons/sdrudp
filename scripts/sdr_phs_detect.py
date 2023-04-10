#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sdrudp import resample
from sdrudp.sdr import SDR

NSAMPLES = 1024
NBLOCKS = 1
SAMPLE_RATE = 2.2e6
LO = 901e6
GAIN = 0
TONE = 901.5e6

PLOT = True
TIMING = False
SAVE = False

t = np.arange(NSAMPLES) / SAMPLE_RATE
omega = 2 * np.pi * (TONE - LO)
tone_cos = np.cos(omega * t)
tone_sin = -np.sin(omega * t)  # conjugate


def on_close(event):
    global running
    running = False


sdr = SDR(direct=False, center_freq=LO, sample_rate=SAMPLE_RATE, gain=GAIN)

if TIMING:
    import time

    SAVE = False
    PLOT = False
    t0 = time.time()

if SAVE:
    all_data = []
    all_rs_data = []
    all_phi = []
    all_rs_phi = []
    all_sample_adc = []

if PLOT:
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_close)
    (line0,) = ax.plot(t, np.ones(NSAMPLES), label="original")
    (line1,) = ax.plot(t, np.ones(NSAMPLES), label="resampled")
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phase [rad]")
    ax.legend(loc="upper right")
    plt.grid()

running = True
sample_adc = SAMPLE_RATE + 1e-7
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
        phi = resample.phase(real, imag, tone_cos, tone_sin)
        dphi = np.mean(resample.diff(phi))
        new_sample_rate = resample.sample_rate_adc(SAMPLE_RATE, omega, dphi)
        if np.abs(dphi) > np.pi / NSAMPLES:
            w = 0
        else:
            w = 0.1
        sample_adc = w * new_sample_rate + (1 - w) * sample_adc
        print(new_sample_rate / 1e6, sample_adc / 1e6, np.abs(dphi))
        rs_real = resample.resample(real, sample_adc, SAMPLE_RATE)
        rs_imag = resample.resample(imag, sample_adc, SAMPLE_RATE)
        rs_phi = resample.phase(rs_real, rs_imag, tone_cos, tone_sin)

        if SAVE:
            all_data.append(data)
            all_rs_data.append(np.array([rs_real, rs_imag]).T)
            all_phi.append(phi)
            all_rs_phi.append(rs_phi)
            all_sample_adc.append(sample_adc)

        if PLOT:
            line0.set_ydata((phi - phi[0] + np.pi) % (2 * np.pi) - np.pi)
            line1.set_ydata((rs_phi - rs_phi[0] + np.pi) % (2 * np.pi) - np.pi)
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        break

sdr.close()
if SAVE:
    d = {
        "nsamples": NSAMPLES,
        "nblocks": NBLOCKS,
        "sample_rate": SAMPLE_RATE,
        "lo": LO,
        "gain": GAIN,
        "tone": TONE,
        "data": all_data,
        "rs_data": all_rs_data,
        "dphi": all_phi,
        "rs_dphi": all_rs_phi,
        "sample_adc": all_sample_adc,
    }
    np.savez("data.npz", **d)
