from bart import bart
import sigpy as sp
import numpy as np


def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_fft():
    img_shape = [256, 256]
    phantom = sp.shepp_logan(img_shape).astype(np.complex128)

    fft_bart = bart(1, 'fft -u 3', phantom)
    fft_sp = sp.fft(phantom, axes=(-2, -1))

    rel_err = rel_l2(fft_bart, fft_sp)
    assert rel_err < 1e-6, f"FFT mismatch: rel_err={rel_err}"