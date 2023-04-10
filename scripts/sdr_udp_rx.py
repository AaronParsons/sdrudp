import socket
import select
import threading
from sdrudp import compress, packet, resample
import numpy as np
import matplotlib.pyplot as plt

PLOT = True
SAMPLE_RATE = 2.2e6
LO = 901e6
TONE = 901.5e6
omega = 2 * np.pi * (TONE - LO)


class UdpRx:
    def __init__(self, host, port=packet.PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(0)
        self._host = (host, port)
        self.prev_mcnt = {}
        self.prev_data = {}

        if PLOT:
            self.init_plot()

    def init_plot(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        self.lines = []
        for ax in self.axs:
            # XXX hardcoded 2048
            (l0,) = self.ax.plot(np.zeros(2048), label="original")
            (l1,) = self.ax.plot(np.zeros(2048), label="resampled")
            self.lines.extend([l0, l1])
        axs[0].legend(loc="upper right")
        plt.setp(axs, ylim=(-np.pi, np.pi))
        plt.grid()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def start(self):
        self._sock.bind(self._host)
        try:
            sample_adc = SAMPLE_RATE + 1e-7
            while True:
                ready = select.select([self._sock], [], [], 0.1)
                if ready[0]:
                    pkt, addr = self._sock.recvfrom(packet.PACKET_SIZE)
                    thd = threading.Thread(
                        target=self.packet_handler, args=(pkt, addr)
                    )
                    thd.start()
                keys = list(self.prev_data.keys())
                if len(keys) < 2:
                    continue
                data = np.array([self.prev_data[k] for k in keys])
                data = data - np.mean(data, axis=1, keepdims=True)

                if PLOT:
                    nsamples = data.shape[1]
                    t = np.arange(nsamples) / SAMPLE_RATE
                    cos = np.cos(omega * t)
                    sin = -np.sin(omega * t)
                    for i, d in enumerate(data):
                        line = self.lines[2*i]
                        phi = resample.phase(d[:, 0], d[:, 1], cos, sin)
                        line.set_ydata(
                            (phi - phi[0] + np.pi) % (2 * np.pi) - np.pi)
                        )
                        if i == 0:
                            dphi = np.mean(resample.diff(phi))
                            new_sample_rate = resample.sample_rate(
                                SAMPLE_RATE, omega, dphi
                            )
                            if np.abs(dphi) > np.pi / nsamples:
                                w = 0
                            else:
                                w = 0.1
                            sample_adc *= 1 - w
                            sample_adc += w * new_sample_rate
                        real = resample.resample(
                            d[:, 0], sample_adc, SAMPLE_RATE
                        )
                        imag = resample.resample(
                            d[:, 1], sample_adc, SAMPLE_RATE
                        )
                        rs_phi = resample.phase(real, imag, cos, sin)
                        line = self.lines[2*i + 1]
                        line.set_ydata(
                            (rs_phi - rs_phi[0] + np.pi) % (2 * np.pi) - np.pi)
                        )
                        self.fig.canvas.restore_region(self.background)
                        self.ax.draw_artist(line)
                        self.fig.canvas.blit(self.ax.bbox)
                        self.fig.canvas.flush_events()

        except KeyboardInterrupt:
            print("Closing...")
            # fig, axes = plt.subplots(nrows=2, sharex=True)
            k0, k1 = list(self.prev_data.keys())
            d0 = self.prev_data[k0].astype(float)
            d1 = self.prev_data[k1].astype(float)
            d0 = d0[:, 0] + 1j * d0[:, 1]
            d1 = d1[:, 0] + 1j * d1[:, 1]
            plt.figure()
            plt.plot(np.angle(d0 * d1.conj()))
            # for k, v in self.prev_data.items():
            #    axes[0].plot(v[:, 0], label=k)
            #    axes[1].plot(v[:, 1], label=k)
            plt.show()
        finally:
            self._sock.close()

    def packet_handler(self, pkt, addr):
        hdr, data = packet.unpack(pkt)
        data = (np.frombuffer(data, dtype="uint8") - 128).view("int8")
        data.shape = (-1, 2)
        # data = compress.decompress(data, nbits=8, signed=True,
        #            mode='cust', bitshuffle=True, gray=True)
        id_num = hdr[0]
        mcnt = hdr[-3]
        if mcnt != self.prev_mcnt.get(id_num, 0) + packet.PAYLOAD_SIZE // 2:
            print(addr, hdr, len(data))
        self.prev_mcnt[id_num] = mcnt
        self.prev_data[id_num] = data


if __name__ == "__main__":
    rx = UdpRx("localhost")
    rx.start()
