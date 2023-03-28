#! /usr/bin/env python
import asyncio
import time
import sys
from sdrudp.sdr import SDR, handle_exception, shutdown
import functools
import signal
import socket
from sdrudp import packet
from sdrudp import compress
import numpy as np

SAMPLE_RATE = 1.8e6
#CENTER_FREQ = 1420e6
CENTER_FREQ = 901e6


async def _streaming(sdr, ip, size=packet.PAYLOAD_SIZE, port=packet.PORT):
    async for d in sdr.stream(num_samples_or_bytes=size, format='bytes'):
        assert len(d) == size
        #d = np.frombuffer(d, dtype='uint8') # not bothering to sign
        #d.shape = (-1, 2)  # I/Q
        #d = d.T.flatten()  # put all I, then all Q
        #d = compress.compress(d, mode='cust', bitshuffle=True, gray=True)
        # step back to beginning of pkt, 2 8-bit IQ numbers per sample
        t_pkt_start_ns = time.time_ns() - int(size//2 * sdr.t_sample_ns)
        sdr.header_args[1] = len(d)
        sdr.header_args[3] = t_pkt_start_ns
        sdr.header_args[-2] += size // 2  # for number I/Q
        pkt = packet.pack_header(*sdr.header_args) + d
        if sdr.is_running:
            sdr.raw_send(pkt, ip, port)
        else:
            break
    try:
        await sdr.stop()
    except(AssertionError):
        pass
    return


class TX_SDR(SDR):
    def __init__(self, device_index=0, direct=False,
                 center_freq=CENTER_FREQ, sample_rate=SAMPLE_RATE):
        super().__init__(device_index=device_index,
                         direct=direct, center_freq=center_freq,
                         sample_rate=sample_rate)
        self.is_running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        id_num = self.dev_p.value  # in case there's more than one sdr
        self.t_sample_ns = 1e9 / self.get_sample_rate()
        lo_hz = self.center_freq
        self.header_args = [id_num, None, self.t_sample_ns, None, 0, lo_hz]

    def stop(self):
        self.is_running = False
        self.sock.close()
        return super().stop()

    def raw_send(self, pkt, ip, port):
        self.sock.sendto(pkt, (ip, port))

    def stream_data(self, ip):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(s,
                    lambda: asyncio.create_task(
                        shutdown(loop, self, signal=s))
                )
            h = functools.partial(handle_exception, sdr=self)
            loop.set_exception_handler(h)
            data = loop.run_until_complete(_streaming(self, ip))
        finally:
            self.is_running = False
            loop.close()


if __name__ == '__main__':
    ip = sys.argv[-1]
    dev_id = int(sys.argv[-2])
    sdr = TX_SDR(device_index=dev_id)
    print(f'Streaming to {ip}:{packet.PORT}')
    try:
        sdr.stream_data(ip)
    except(RuntimeError):
        pass
