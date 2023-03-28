'''Module for compressing ADC integers.'''
# NOTE: removed -2**(nbits-1) as a supported number

import numpy as np
import lz4.frame  # only for lz4 encoding

def _gray(b, nbits, signed=True):
    '''Encode data with a simple gray encoding to avoid many bits
    changing between adjacent numbers.'''
    if signed:
        # offset signed to unsigned int
        ab = (b + 2**(nbits-1)).astype(f'uint{nbits}')
    else:
        ab = b
    g = ab ^ (ab >> 1)  # bitshift xor for gray encoding
    # put sign bit on bottom and flip polarity of next highest bit to
    # ensure maximal zeroes in top bits
    g = ((g ^ (1<<(nbits-2))) << 1) | (g >> (nbits-1))
    return g


def _degray(g, nbits, signed=True):
    '''Decode data from gray encoding.'''
    # undo sign/polarity bit swapping
    g = ((g & 1) << (nbits-1)) | ((g ^ (1<<(nbits-1))) >> 1)
    # undo bitshift xor gray encoding
    for i in range(int(np.log2(nbits))):
        g ^= (g >> 2**i)
    if signed:
        # recast from unsigned to signed
        b = g.view(f'int{nbits}')
        b -= 2**(nbits-1)
    return b
    

def _get_used_bits(b, nbits):
    '''Return number of bits used to represent values in array b.'''
    m = np.bitwise_or.reduce(b).view(f'uint{nbits}')
    return int(np.ceil(np.log2(m + 1)))


def _bitshuffle(d, nbits):
    '''Apply bitshuffle bitwise transpose of nbits numbers.'''
    scalar = 2**np.arange(nbits)
    scalar.shape = (1, nbits)
    out = []
    for i in range(nbits):
        buf = (d >> (nbits - i -1)) & 1
        buf.shape = (-1, nbits)
        buf = np.sum(buf * scalar, axis=1, keepdims=True)
        out.append(buf)
    out = np.concatenate(out, axis=0).flatten()
    return out.astype(f'uint{nbits}')


def _debitshuffle(d, nbits, dtype='uint'):
    '''Unapply bitshuffle bitwise transpose of nbits numbers.'''
    out = []
    scalar = 2**np.arange(nbits)[::-1]
    scalar.shape = (1, nbits)
    stride = d.shape[0] // nbits
    for b in [d[j::stride] for j in range(stride)]:
        for i in range(nbits):
            buf = (b >> i) & 1
            buf.shape = (-1, nbits)
            buf = np.sum(buf * scalar, axis=1, keepdims=True)
            out.append(buf)
    out = np.concatenate(out, axis=0).flatten()
    return out.astype(f'{dtype}{nbits}')


def _rot_encode(d, rotm, rescale=2**5, dtype_out=None):
    if dtype_out is None:
        nbits = d.dtype.itemsize * 8
        dtype_out = f'int{nbits*2}'
    d = np.around(np.dot(d.reshape(-1, rotm.shape[0]), rotm) / rescale).astype(dtype_out)
    return d.T.flatten()


def _rot_decode(d, irotm, rescale=2**5):
    d = np.around(np.dot(d.reshape(irotm.shape[0], -1).T, irotm) * rescale).flatten()
    return d


def _block_compress(d, nbits, signed=True,
                    gray=True, bitshuffle=True, mode='cust'):
    '''Apply compression (mode in ['lz4', 'cust']) w/ gray/bitshuffle
    encoding for provided array block.'''
    if gray:
        d = _gray(d, nbits, signed=signed)
    if mode == 'cust':
        assert bitshuffle
        assert gray
        m = _get_used_bits(d, nbits)
        d = np.pad(_bitshuffle(d, nbits), (1, 0))[-m * d.shape[0] // nbits - 1:]
        d[0] = m
    else:
        if bitshuffle:
            d = _bitshuffle(d, nbits)
    return d
        
        
def _block_decompress(d, nbits, signed, block,
                      gray=True, bitshuffle=True, mode='cust'):
    '''Unapply compression (mode in ['lz4', 'cust']) w/ gray/bitshuffle
    encoding for provided compressed array block.'''
    if mode == 'cust':
        assert bitshuffle
        d = np.pad(d, (block-d.shape[0], 0))
    elif mode == 'lz4':
        #d = lz4.frame.decompress(d)
        pass  # decode later
    else:
        raise ValueError(f'Unrecognized mode: {mode}')
    if bitshuffle:
        d = _debitshuffle(d, nbits)
    if gray:
        d = _degray(d, nbits, signed=signed)
    return d
        
        
def _split_cust_blocks(d, nbits, block):
    '''Generator for splitting an cust-compressed array back into blocks.'''
    i = 0
    while i < d.shape[0]:
        m = d[i]
        j = i + 1 + m * block // nbits
        yield d[i+1:j]
        i = j
    return


def compress(d, nbits=None, gray=True, bitshuffle=True, mode='lz4',
             rotm=None, block=256, rescale=2**6, dtype_out=None):
    '''Apply specified compression to array 'd'.'''
    if nbits is None:
        nbits = 8 * d.dtype.itemsize
    if rotm is not None:
        if dtype_out is None:
            dtype_out = f'int{2*nbits}'
            nbits *= 2
        d = _rot_encode(d, rotm, rescale=rescale).astype(dtype_out)
    signed = bool(d.dtype.kind == 'i')
    buf = [_block_compress(d[i:i+block], nbits, signed=signed, gray=gray,
                           bitshuffle=bitshuffle, mode=mode)
           for i in range(0, d.shape[0], block)]
    buf = np.concatenate(buf)
    if mode == 'lz4':
        return lz4.frame.compress(buf)
    elif mode == 'cust':
        return buf.tobytes()
    else:
        raise ValueError(f'Unrecognized mode: {mode}')
    

def decompress(d, nbits, signed, gray=True, bitshuffle=True, mode='lz4',
               irotm=None, block=256, rescale=2**6, dtype_out=None):
    '''Unapply specified compression to array 'd'.'''
    dtype = f'uint{nbits}'
    # XXX could be int if lz4 and not gray or bitshuffle
    if dtype_out is None:
        dtype_out = f'int{nbits}'
    if mode == 'lz4':
        d = np.frombuffer(lz4.frame.decompress(d), dtype=dtype).copy()
        buf = [d[i:i+block] for i in range(0, d.shape[0], block)]
    elif mode == 'cust':
        d = np.frombuffer(d, dtype=dtype)
        buf = _split_cust_blocks(d, nbits, block)
    else:
        raise ValueError(f'Unrecognized mode: {mode}')
    d = np.concatenate([_block_decompress(b, nbits, signed, block,
                                          gray=gray, bitshuffle=bitshuffle, mode=mode)
                        for b in buf])
    if irotm is not None:
        d = _rot_decode(d, irotm, rescale=rescale)
    d = d.astype(dtype_out)
    return d


def est_rot_matrix(n, dim=8, rescale=2**6, dtype='int16'):
    '''Estimate eigenmode proj/deproj matrices from provided array.
    Rounds proj matrix to an integer array; deproj is a float array
    representing the true inverse of the rounded proj array.'''
    n = n[:n.size - n.size % dim].reshape(-1, dim).astype(float)
    cov = np.dot(n.T, n)
    U, _, _ = np.linalg.svd(cov, hermitian=True)
    Ur = np.around(U * rescale)
    iUr = np.linalg.inv(Ur)
    return Ur.astype(dtype), iUr
    

def gen_noise(scale, cen=0, size=1024, dtype='int8'):
    '''Generate quantized normal noise.'''
    nbits = 8 * np.dtype(dtype).itemsize
    n = np.random.normal(loc=cen, scale=scale, size=size)
    # remove -2**(nbits-1) as a supported number
    n = np.around(n).clip(-2**(nbits-1)+1, 2**(nbits-1)-1).astype(dtype)
    return n


def gen_sine(scale, period, cen=0, size=1024, phs0=1, dtype='int8'):
    '''Generate quantized sine wave.'''
    nbits = 8 * np.dtype(dtype).itemsize
    n = scale * np.sin(2*np.pi * np.arange(size) / period + phs0) + cen
    # remove -2**(nbits-1) as a supported number
    n = np.around(n).clip(-2**(nbits-1)+1, 2**(nbits-1)-1).astype(dtype)
    return n


def calc_size(b):
    '''Calculate the size in bytes of an array, string, or bytes.'''
    if type(b) in (str, bytes):
        return len(b)
    else:
        return b.size * b.dtype.itemsize
    
    
def view_binimg(d):
    '''Return an array where each row represnts bits from d.'''
    nbits = d.dtype.itemsize * 8
    bits = np.array([(d >> i) & 1 for i in range(nbits)], dtype=bool)
    return bits
