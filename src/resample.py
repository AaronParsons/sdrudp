import numpy as np
from scipy.signal import get_window

RES = 0.001
DTYPE = np.float32
ndt = np.concatenate(
    [np.array([0], dtype=DTYPE), np.arange(1, 0, -RES, dtype=DTYPE)]
)

_cache = {}


def _clear_cache():
    global _cache
    _cache = {}


def _cache_filter(window, half_width, dtype):
    key = (window, half_width)
    if key not in _cache:
        x = np.arange(-half_width, half_width + 1, dtype=dtype)
        win = get_window(window, 2 * half_width + 1).astype(dtype)
        sinc = win[:, None] * np.sinc(x[:, None] - ndt[None, :])
        _cache[key] = sinc
    return _cache[key]


def _cache_ker(nsamples, f_in, f_out, dt0, window, half_width, dtype):
    key = (nsamples, f_in, f_out, dt0, window, half_width)
    if key not in _cache:
        x = np.arange(-half_width, half_width + 1, dtype=int)
        i = np.arange(nsamples, dtype=int)
        dt = 1 - f_in / f_out
        dt = np.arange(dt0, dt0 + nsamples * dt, dt, dtype=dtype)
        off = np.ceil(dt).astype(int)
        dt -= off
        i -= off
        inds = (x[:, None] + i[None, :]).clip(0, nsamples - 1)
        sinc = _cache_filter(window, half_width, dtype)
        ker = sinc[:, np.around(dt / RES).astype(int)]
        _cache[key] = inds, ker
    return _cache[key]


def resample(
    sig_in, f_in, f_out, dt0=0, window="nuttall", half_width=128, dtype=DTYPE
):
    """
    Resample a signal using sinc interpolation.

    Parameters
    ----------
    sig_in : ndarray
        Input signal with shape ([nblocks,] nsamples).
    f_in : float
        Input sampling frequency.
    f_out : float
        Output sampling frequency.
    dt0 : float, optional
        Initial time offset.
    window : str, float, or tuple.
        Window function for sinc interpolation. See `scipy.signal.get_window`
        for details.
    half_width : int
        Half-width of the interpolation window.
    dtype : dtype
        Data type of the output signal.

    Returns
    -------
    sig_out : ndarray
        The resampled signal.
    """
    nsamples = sig_in.shape[-1]
    inds, ker = _cache_ker(
        nsamples, f_in, f_out, dt0, window, half_width, dtype
    )
    # ker[inds < 0] = 0  # boundary is messed up anyway
    # ker[inds >= nsamples] = 0  # boundary is messed up anyway
    sig_out = np.einsum("...ij,ij->...j", sig_in[..., inds], ker)
    return sig_out
    #return np.sum(sig_in[:, inds] * ker, axis=-2)  # slightly slower than einsum
