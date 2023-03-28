import struct

VERSION = 1
PORT = 8081
STRUCT_STR = 'LHHLfQQdQ'
HEADER_SIZE = struct.calcsize(STRUCT_STR)
PAYLOAD_SIZE = 4096
PACKET_SIZE = HEADER_SIZE + PAYLOAD_SIZE

def pack_header(id_num, payload_size,
                t_sample_ns, t_pkt_start_ns, mcnt, lo_hz):
    hdr = struct.pack(STRUCT_STR,
                      id_num, HEADER_SIZE, payload_size,
                      VERSION, t_sample_ns,
                      t_pkt_start_ns,
                      mcnt,
                      lo_hz,
                      0)
    return hdr

def unpack(packet):
    hdr = struct.unpack(STRUCT_STR, packet[:HEADER_SIZE])
    data = packet[HEADER_SIZE:]
    return hdr, data
