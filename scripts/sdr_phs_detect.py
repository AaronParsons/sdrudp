import numpy as np
import asyncio
import time
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import logging
import signal
from ugradio.sdr import SDR
import hera_filters

NSAMPLES = 2048
NBLOCKS = 40
SAMPLE_RATE = 3.2e6
LO = 901e6
GAIN = 0
TONE = 901.5e6

PLOT = False

tone_cos = np.cos(2 * np.pi * (TONE - LO) * np.arange(NSAMPLES) / SAMPLE_RATE)
tone_sin = -np.sin(2 * np.pi * (TONE - LO) * np.arange(NSAMPLES) / SAMPLE_RATE)  # conjugate


running = True
def on_close(event):
    global running
    running = False

if PLOT:
    plt.ion()
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.canvas.mpl_connect('close_event', on_close)
    line0, = ax.plot(np.ones(NSAMPLES))
    line1, = ax.plot(np.ones(NSAMPLES))
    #line2, = ax.plot(np.ones(NSAMPLES))
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.ylim(-128, 128)
    plt.grid()

sdr = SDR(direct=False, center_freq=LO, sample_rate=SAMPLE_RATE, gain=GAIN)
           
t0 = time.time()
while running:
    try:
        data = sdr.capture_data(nsamples=NSAMPLES, nblocks=NBLOCKS)
        t1 = time.time()
        print((t1 - t0) / (NSAMPLES  * NBLOCKS / SAMPLE_RATE))
        t0 = t1
        if PLOT:
            data = data - np.mean(data, axis=1, keepdims=True)  # remove DC offset
            data_cos = data[0,...,0] * tone_cos - data[0,...,1] * tone_sin
            data_sin = data[0,...,0] * tone_sin + data[0,...,1] * tone_cos
            line0.set_ydata(data_cos)
            line1.set_ydata(data_sin)
            #dt = np.arctan2(data_cos, data_sin) / (2 * np.pi * (TONE - LO) / SAMPLE_RATE)
            #line2.set_ydata(dt / np.pi * 128)
            #print(np.mean(np.diff(dt)[:500]))
            fig.canvas.draw()
            fig.canvas.flush_events()
    except(KeyboardInterrupt):
        break
