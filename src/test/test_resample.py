import numpy as np
from .. import compress, resample

np.random.seed(0)


def test_resample():
    # check that we can resample blocks of data
    f_in = 3.19e6
    f_out = 3.20e6

    dtype = "int8"
    nsamples = 2048
    nblocks = 500
    noise = compress.gen_noise(0.1, size=(nblocks, nsamples), dtype=dtype)
    sine = compress.gen_sine(30, 4.37, cen=0, size=nsamples, dtype=dtype)
    x = sine[None] + noise
    y_loop = np.empty_like(x)
    for i in range(nblocks):
        y_loop[i] = resample.resample(x[i], f_in, f_out, dtype=dtype)
    y = resample.resample(x, f_in, f_out, dtype=dtype)
    assert np.allclose(y, y_loop)
