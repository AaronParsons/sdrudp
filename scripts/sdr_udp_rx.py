import packet
import socket
import select
import threading
import time
import compress
import numpy as np
import matplotlib.pyplot as plt

class UdpRx:

    def __init__(self, host, port=packet.PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(0)
        self._host = (host, port)
        self.prev_mcnt = {}
        self.prev_data = {}

    def start(self):
        self._sock.bind(self._host)
        try:
            while True:
                ready = select.select([self._sock], [], [], .1) 
                if ready[0]:
                    pkt, addr = self._sock.recvfrom(packet.PACKET_SIZE)
                    thd = threading.Thread(target=self.packet_handler, args=(pkt, addr))
                    thd.start()
        except(KeyboardInterrupt):
            print('Closing...')
            #fig, axes = plt.subplots(nrows=2, sharex=True)
            k0, k1 = list(self.prev_data.keys())
            d0 = self.prev_data[k0].astype(float)
            d1 = self.prev_data[k1].astype(float)
            d0 = d0[:,0] + 1j * d0[:,1]
            d1 = d1[:,0] + 1j * d1[:,1]
            plt.plot(np.angle(d0 * d1.conj()))
            #for k, v in self.prev_data.items():
            #    axes[0].plot(v[:, 0], label=k)
            #    axes[1].plot(v[:, 1], label=k)
            plt.show()
        finally:
            self._sock.close()

    def packet_handler(self, pkt, addr):
        hdr, data = packet.unpack(pkt)
        data = (np.frombuffer(data, dtype='uint8') - 128).view('int8')
        data.shape = (-1, 2)
        #data = compress.decompress(data, nbits=8, signed=True, 
        #            mode='cust', bitshuffle=True, gray=True)
        id_num = hdr[0]
        mcnt = hdr[-3]
        if mcnt != self.prev_mcnt.get(id_num, 0) + packet.PAYLOAD_SIZE // 2:
            print(addr, hdr, len(data))
        self.prev_mcnt[id_num] = mcnt
        self.prev_data[id_num] = data


if __name__ == '__main__':
    rx = UdpRx('localhost')
    rx.start()
