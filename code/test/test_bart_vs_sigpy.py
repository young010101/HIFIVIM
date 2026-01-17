from bart import bart
import sigpy as sp
import numpy as np
import sigpy.mri as mr
import matplotlib.pyplot as plt

img_shape = [256, 256]
num_coils = 8
phantom = sp.shepp_logan(img_shape).astype(np.complex128)


def rel_l2(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_fft():

    fft_bart = bart(1, 'fft -u 3', phantom)
    fft_sp = sp.fft(phantom, axes=(-2, -1))

    rel_err = rel_l2(fft_bart, fft_sp)
    assert rel_err < 1e-6, f"FFT mismatch: rel_err={rel_err}"


def test_espirit_calib():
    Nx, Ny = img_shape
    Nc = num_coils
    
    # 3. Simulate "Birdcage" coil sensitivity maps
    # These simulate physical coil profiles
    mps_true = mr.birdcage_maps((num_coils, *img_shape))

    # 4. Generate k-space data (Image * Sensitivity -> FFT)
    # This mimics raw data coming off a scanner
    ksp = sp.fft(phantom * mps_true, axes=(-2, -1))

    # 5. Estimate sensitivity maps from k-space using ESPIRiT
    # calib_width defines the central 'auto-calibration' region (ACS)
    app = mr.app.EspiritCalib(ksp, calib_width=24, device=sp.cpu_device)
    mps_estimated = app.run()

    # --- BART ecalib ---
    ksp_bart = np.moveaxis(ksp, 0, -1)  # BART expects coils last
    ksp_bart_expand = ksp_bart[:,:,None, :]
    # mps_estimated_bart = bart(1, 'ecalib -m1', ksp_bart_expand)
    mps_estimated_bart = bart(1, 'ecalib -m1 -r 24 -k 6 -c 0.95 -t 0.02', ksp_bart_expand)

    mps_estimated_bart = np.moveaxis(mps_estimated_bart, -1, 0)[...,0] # Move coils to first dim, remove extra dim

    mps_sigpy_xycm = to_xycm(mps_estimated, Nx, Ny, Nc)
    mps_bart_xycm  = to_xycm(mps_estimated_bart, Nx, Ny, Nc)
    assert mps_sigpy_xycm.shape == mps_bart_xycm.shape, "ESPIRiT Calib shape mismatch"
    # assert mps_estimated.shape == mps_estimated_bart.shape, "ESPIRiT Calib shape mismatch"

    P_sigpy = proj_from_maps(mps_sigpy_xycm)
    P_bart  = proj_from_maps(mps_bart_xycm)

    err = rel_l2(P_sigpy, P_bart)
    assert err < 5e-2, f"ESPIRiT subspace mismatch (projection): rel_err={err}"

    if False:
        show_imgs(abs(mps_estimated))
        show_imgs(abs(mps_estimated_bart))
    # err = rel_l2(mps_estimated, mps_estimated_bart)
    # assert err < 1e-6, f"ESPIRiT Calib mismatch: rel_err={err}"


def show_imgs(imgs, ncols=None, cmap='gray', titles=None, figsize=(10,10)):
    imgs = np.asarray(imgs)
    N = imgs.shape[0]
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < N:
            ax.imshow(imgs[i], cmap=cmap)
            if titles:
                ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def to_xycm(mps, Nx, Ny, Nc):
    mps = np.asarray(mps)
    # SigPy: (Nc,Nx,Ny) -> (Nx,Ny,Nc,1)
    if mps.shape == (Nc, Nx, Ny):
        mps = np.moveaxis(mps, 0, -1)          # (Nx,Ny,Nc)
        return mps[..., None]                  # (Nx,Ny,Nc,1)

    # BART typical: (Nx,Ny,1,Nc,1,1) or similar
    if mps.ndim == 6 and mps.shape[0] == Nx and mps.shape[1] == Ny and mps.shape[3] == Nc:
        return mps[:, :, 0, :, 0, :]           # (Nx,Ny,Nc,Nm)
    elif mps.ndim == 4 and mps.shape[0] == Nx and mps.shape[1] == Ny and mps.shape[3] == Nc:
        return mps[:, :, 0, :, None]           # (Nx,Ny,Nc,Nm)

    # If already (Nx,Ny,Nc) or (Nx,Ny,Nc,Nm)
    if mps.shape == (Nx, Ny, Nc):
        return mps[..., None]
    if mps.shape[0] == Nx and mps.shape[1] == Ny and mps.shape[2] == Nc:
        return mps

    raise ValueError(mps.shape)


def proj_from_maps(mps_xycm, lam=1e-6):
    # mps: (Nx,Ny,Nc,Nm)
    Nx, Ny, Nc, Nm = mps_xycm.shape
    S = mps_xycm.reshape(-1, Nc, Nm).astype(np.complex64)

    P = np.zeros((S.shape[0], Nc, Nc), dtype=np.complex64)
    I = np.eye(Nm, dtype=np.complex64)

    for i in range(S.shape[0]):
        Si = S[i]                              # (Nc,Nm)
        G = Si.conj().T @ Si + lam * I         # (Nm,Nm)
        X = np.linalg.solve(G, Si.conj().T)    # (Nm,Nc)
        P[i] = Si @ X                          # (Nc,Nc)

    return P.reshape(Nx, Ny, Nc, Nc)

