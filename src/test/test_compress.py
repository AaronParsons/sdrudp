import unittest
import numpy as np
import compress
import matplotlib.pyplot as plt

np.random.seed(0)

class TestCompress(unittest.TestCase):

    def test_gray(self):
        nbits = 8
        d = np.arange(-127, 127)
        g = compress._gray(d, nbits)
        assert np.unique(g).size == d.size
        bits = np.bitwise_xor(g[:-1], g[1:])
        assert np.all(bits == 2**np.around(np.log2(bits)).astype(int))

    def test_degray(self):
        for nbits in (8, 16, 32):
            scale = min(2**(nbits-1), 2**23)
            d = np.arange(-(scale-1), scale, dtype=f'int{nbits}')
            g = compress._gray(d, nbits)
            ig = compress._degray(g, nbits)
            assert np.all(d == ig)

    def test_get_used_bits(self):
        assert 1 == compress._get_used_bits(np.array([0,1], dtype='uint8'), 8)
        assert 2 == compress._get_used_bits(np.array([2], dtype='uint8'), 8)
        assert 2 == compress._get_used_bits(np.array([2], dtype='uint16'), 16)
        assert 7 == compress._get_used_bits(np.array([64, 127], dtype='uint8'), 8)

    def test_bitshuffle(self):
        d = np.array([0] * 8, dtype='int8')
        b = compress._bitshuffle(d, 8)
        assert b.dtype == np.dtype('uint8')
        assert np.all(b == 0)
        d = np.array([1] * 8, dtype='int8')
        b = compress._bitshuffle(d, 8)
        assert np.all(b[:7] == 0)
        assert np.all(b[7] == 255)
        d = np.array([-1] * 32, dtype='int16')
        b = compress._bitshuffle(d, 16)
        assert b.dtype == np.dtype('uint16')
        assert np.all(b == 2**16 - 1)

    def test_debitshuffle(self):
        for nbits in (8, 16, 32):
            scale = min(2**(nbits-1), 2**15)
            d = np.arange(-scale, scale, dtype=f'int{nbits}')
            b = compress._bitshuffle(d, nbits)
            ib = compress._debitshuffle(b, nbits, dtype='int')
            assert ib.dtype == np.dtype(f'int{nbits}')
            assert np.all(d == ib)

    def test_end_to_end(self):
        dtype = 'int8'
        nbits = 8 * np.dtype(dtype).itemsize
        SIZE = 2048
        BLOCK = 256
        RESCALE = 2**5  # can fail every so often, dep on random seed
        noise = compress.gen_sine(30, period=4.37, cen=0, size=SIZE, dtype=dtype) \
                + compress.gen_noise(1.1, size=SIZE, dtype=dtype)
        rotm, irotm = compress.est_rot_matrix(noise, dim=8)
        nr = compress.compress(noise,
                               rotm=rotm, block=BLOCK, rescale=RESCALE,
                               mode='cust', bitshuffle=True, gray=True)
        factor = compress.calc_size(nr) / compress.calc_size(noise)
        nd = compress.decompress(nr, 2*nbits, True,
                                 irotm=irotm, block=BLOCK, rescale=RESCALE, dtype_out=dtype,
                                 mode='cust', bitshuffle=True, gray=True)
        assert factor < 0.7
        assert np.all(nd == noise)

if __name__ == '__main__':
    unittest.main()
        
