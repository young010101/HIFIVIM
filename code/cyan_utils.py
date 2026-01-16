import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from scipy.optimize import least_squares
from scipy.special import erf
from utils import dtd_gamma_model, BVALS
from matplotlib.pyplot import plt


def show_15_bvals(x, cmap='gray', low_p = 1, high_p = 99, bvals=BVALS):
    print(x.shape)
    fix, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes_flat = axes.ravel()
    last_im = None
    for i, ax in enumerate(axes_flat):
        vmin, vmax = np.percentile(np.abs(x[:, :, i]), [low_p, high_p])
        last_im = ax.imshow(np.abs(x[:, :, i]), cmap=cmap, vmin=vmin, vmax=vmax)
        if bvals is not None:
            ax.set_title(f'b_val = {bvals[i]}')
        ax.axis('off')
    # Add a single shared colorbar for the grid
    if last_im is not None:
        cax = fix.add_axes([axes[-1, -1].get_position().x1 + 0.1,
                           axes[-1, -1].get_position().y0,
                           0.01,
                           axes[0, -1].get_position().y1 - axes[-1, -1].get_position().y0])
        fix.colorbar(last_im, cax=cax)
    plt.tight_layout()
    plt.show()


def make_phantom(args, show=True, slice=90, points=None, bvals=BVALS):
    import nibabel as nib
    import matplotlib.pyplot as plt

    phantom = nib.load(os.path.join(args.outdir, 'Phantom_T1.nii.gz')).get_fdata()[..., slice]
    phantom[phantom < 200] = 0
    phantom = phantom / phantom.max() * 4.  # scale to 0-3.

    md, vi, va = np.zeros_like(phantom), np.zeros_like(phantom), np.zeros_like(phantom)
    phantom_data = np.zeros((phantom.shape[0], phantom.shape[1], bvals.shape[0]))
    phantom_data_b_delta_1 = np.zeros_like(phantom_data)
    WMmask, GMmask, CSFmask, WMH_mask1,WMH_mask2, BGmask = np.zeros_like(md), np.zeros_like(md), \
                                              np.zeros_like(md), np.zeros_like(md), np.zeros_like(md), np.zeros_like(md)

    center = (45,100)
    height, width = phantom.shape

    WMH_mask1 = np.zeros_like(md)
    y, x = np.ogrid[:height, :width]
    radius = 5
    WMH_mask1[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

    WMH_mask2 = np.zeros_like(md)
    center, radius = (53,60), 3.5
    WMH_mask2[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

    WMH_mask3 = np.zeros_like(md)
    center, radius = (60,100), 3
    WMH_mask3[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1


    for i in range(phantom.shape[0]):
        for j in range(phantom.shape[1]):
            if WMH_mask1[i,j] == 1:
                # md is stored in μm²/ms units; here md = 1.2 (≈ 0.0012 mm²/s)
                md[i, j] = 1.2
                vi[i, j] = 0.96
                va[i, j] = 0.925
            elif WMH_mask2[i, j] == 1:
                md[i, j] = 1.4
                vi[i, j] = 0.97
                va[i, j] = 0.928
            elif WMH_mask3[i, j] == 1:
                md[i, j] = 1.3
                vi[i, j] = 0.965
                va[i, j] = 0.927
            else:
                if j < 101 and j > 65 and i > 62 and i < 103:
                    md[i, j] = 0.6 if phantom[i, j] > 2.5 else 0.5 if phantom[i, j] > 1.7 else 3 if phantom[
                                                                                                                  i, j] > 0 else 0
                    vi[i, j] = 0.7 if phantom[i, j] > 2.5 else 0.6 if phantom[i, j] > 1.7 else 2.5 if phantom[
                                                                                                            i, j] > 0 else 0
                    va[i, j] = 0.45 if phantom[i, j] > 2.5 else 0.55 if phantom[i, j] > 1.7 else 0.2 if phantom[ i, j] > 0 else 0
                else:
                    md[i, j] = 0.6 if phantom[i, j] > 2.35 else 0.9 if phantom[i, j] > 1.7 else 3 if phantom[
                                                                                                                  i, j] > 0 else 0
                    vi[i, j] = 0.7 if phantom[i, j] > 2.35 else 1.4 if phantom[i, j] > 1.7 else 2 if phantom[
                                                                                                            i, j] > 0 else 0
                    va[i, j] = 0.45 if phantom[i, j] > 2.35 else 0.3 if phantom[i, j] > 1.7 else 0.2 if phantom[
                                                                                                              i, j] > 0 else 0
            if np.isclose(md[i,j], 0.6):
                WMmask[i,j] = 1
            elif np.isclose(md[i, j], 0.5):
                BGmask[i, j] = 1
            elif np.isclose(md[i, j], 0.9):
                GMmask[i, j] = 1
            elif np.isclose(md[i, j], 3):
                CSFmask[i, j] = 1


            phantom_data[i,j] = dtd_gamma_model(10, md[i,j], vi[i,j], va[i,j], bvals)
            phantom_data[i,j][md[i,j] == 0] =0
            phantom_data_b_delta_1[i,j] = dtd_gamma_model(10, md[i,j], vi[i,j], va[i,j], bvals, b_delta=np.ones_like(bvals))
            phantom_data_b_delta_1[i,j][md[i,j] == 0] =0


    if show:
        plt.figure()
        plt.subplot(231), plt.imshow(np.rot90(WMmask), cmap='gray'), plt.axis('off')
        plt.subplot(232),  plt.imshow(np.rot90(GMmask), cmap='gray'), plt.axis('off')
        plt.subplot(233), plt.imshow(np.rot90(BGmask), cmap='gray'), plt.axis('off')
        plt.subplot(234), plt.imshow(np.rot90(WMH_mask1), cmap='gray'), plt.axis('off')
        plt.subplot(235), plt.imshow(np.rot90(WMH_mask2), cmap='gray'), plt.axis('off')
        plt.subplot(236), plt.imshow(np.rot90(WMH_mask3), cmap='gray'), plt.axis('off')

        cmap = 'turbo'
        fig, axes = plt.subplots(1, 4, figsize=(16,4))
        axes[0].imshow(np.rot90(phantom), cmap='gray'), axes[0].set_xticks([]), axes[0].set_yticks([])
        axes[0].set_title('Phantom', fontsize=16, fontweight='bold')
        plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)

        im = axes[1].imshow(np.rot90(md), cmap=cmap)
        axes[1].set_xticks([]), axes[1].set_yticks([])
        axes[1].set_title('MD', fontsize=16, fontweight='bold'), im.set_clim(0.3, 1.5)
        cax = fig.add_axes([axes[1].get_position().x1 + 0.005,
                            axes[1].get_position().y0, 0.01, axes[1].get_position().height])
        cbar = plt.colorbar(axes[1].images[0], cax=cax)


        im = axes[2].imshow(np.rot90(vi), cmap=cmap)
        axes[2].set_xticks([]), axes[2].set_yticks([])
        axes[2].set_title('$V_I$', fontsize=16, fontweight='bold'), im.set_clim()
        cax = fig.add_axes([axes[2].get_position().x1 + 0.005,
                            axes[2].get_position().y0, 0.01, axes[2].get_position().height])
        cbar = plt.colorbar(axes[2].images[0], cax=cax)

        im = axes[3].imshow(np.rot90(va), cmap=cmap)
        axes[3].set_xticks([]), axes[3].set_yticks([])
        axes[3].set_title('$V_A$', fontsize=16, fontweight='bold'), im.set_clim()
        cax = fig.add_axes([axes[3].get_position().x1 + 0.005,
                            axes[3].get_position().y0, 0.01, axes[3].get_position().height])
        cbar = plt.colorbar(axes[3].images[0], cax=cax)

        plt.subplots_adjust(wspace=0.5)

        # ####################################################
        plt.figure(figsize=(15,3))
        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.imshow(np.rot90(phantom_data[...,(i+3)*2]), cmap='gray'), plt.clim(), plt.axis('off')
            plt.title("b = {} s/mm$^2$".format(bvals[(i+3)*2]), fontsize=14, fontweight='bold')
            plt.colorbar()

        plt.show()

        # ########### select some point to plot signal curve ########################
        if points:
            fig, axes = plt.subplots(1, len(points), figsize=(15,3))
            for idx, (i, j) in enumerate(points):
                ax = axes[idx]
                ax.plot(bvals, phantom_data[i, j], 'o-')
                ax.plot(bvals, phantom_data_b_delta_1[i, j], 'x--')
                ax.set_xlabel('b-values (s/mm$^2$)', fontsize=14, fontweight='bold')
                if idx == 0:
                    ax.set_ylabel('Signal Intensity', fontsize=14, fontweight='bold')
                ax.set_title('({},{})'.format(i, j), fontsize=14, fontweight='bold')
                ax.grid()
        

    return md, vi, va, phantom_data, phantom_data_b_delta_1, (np.rot90(WMmask), np.rot90(GMmask), np.rot90(CSFmask),
                                      np.rot90(BGmask), np.rot90(WMH_mask1), np.rot90(WMH_mask2), np.rot90(WMH_mask3))


@DeprecationWarning
def make_toy_sens_2d(nx=128, ny=128, ncoils=32, ring_radius=0.8, eps=1e-3,
                    phase_alpha=2.0,  # 0.0 -> purely real (no extra phase)
                    amp_model="inv_r",  # "inv_r" or "gauss"
                    sigma=0.35,         # used if amp_model="gauss"
                    normalize_sos=True,
                    dtype=np.complex64):
    """
    Returns sens maps with shape (nx, ny, ncoils) complex.
    Coordinates are normalized to [-1,1].
    """
    x = np.linspace(-1, 1, nx, dtype=np.float32)
    y = np.linspace(-1, 1, ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")  # shape (nx, ny)

    sens = np.zeros((nx, ny, ncoils), dtype=np.complex64)

    for k in range(ncoils):
        theta = 2.0 * np.pi * k / ncoils
        xk = ring_radius * np.cos(theta)
        yk = ring_radius * np.sin(theta)

        dx = X - xk
        dy = Y - yk

        if amp_model == "inv_r":
            A = 1.0 / np.sqrt(dx * dx + dy * dy + eps)
        elif amp_model == "gauss":
            A = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
        else:
            raise ValueError("amp_model must be 'inv_r' or 'gauss'")

        # smooth coil-dependent phase (optional)
        phi = phase_alpha * (X * np.cos(theta) + Y * np.sin(theta))
        sens[..., k] = A * np.exp(1j * phi)

    if normalize_sos:
        sos = np.sqrt(np.sum(np.abs(sens) ** 2, axis=-1, keepdims=True) + 1e-12)
        sens = sens / sos

    return sens.astype(dtype)


def bartize_sens(sens_xyc: np.ndarray):
    """
    Convert (nx, ny, coils) -> BART-friendly dims.
    BART convention often uses [X Y Z COIL ...] with COIL at dim=3 (0-based).
    Here we output shape (nx, ny, 1, ncoils) so coil dim is the 4th dimension.
    """
    nx, ny, nc = sens_xyc.shape
    return sens_xyc.reshape(nx, ny, 1, nc)


def ifft2c(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x), axes=(0,1),norm=None))


def plot_points_on_image(img, points, colors=None, title="Points on image"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(np.abs(img), cmap='gray')
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(points)))
    for (i, j), c in zip(points, colors):
        plt.plot(j, i, marker='o', color=c, markersize=6)
        plt.text(j + 2, i - 2, f"({i},{j})", color=c, fontsize=8)
    plt.title(title)
    plt.axis('off')
    plt.show()

# ----------------------------
# Forward model (must match your earlier safe version if needed)
# ----------------------------
def dtd_gamma_fit2data(m: np.ndarray, xps: Dict[str, Any]) -> np.ndarray:
    """
    Equivalent of dtd_gamma_1d_fit2data(m, xps)

    m = [s0, d_iso, mu2_iso, mu2_aniso, (optional sw...)]
    xps needs: b, b_delta, b_eta, s_ind (optional)
    """
    m = np.asarray(m, float)
    s0, d_iso, mu2_iso, mu2_aniso = m[:4]

    b = np.asarray(xps["b"], float)  # SI in MATLAB version (s/m^2) typically
    b_delta = np.asarray(xps.get("b_delta", np.ones_like(b)), float)
    b_eta = np.asarray(xps.get("b_eta", np.zeros_like(b)), float)
    s_ind = np.asarray(xps.get("s_ind", np.ones_like(b, dtype=int)), int)

    # rs is not explicitly in m here; MATLAB uses m(5:end) as sw scaling directly (relative S0 series)
    # In dtd_gamma_1d_fit2data.m: rs = [1 m(5:end)]
    if m.size > 4:
        rs = np.concatenate([[1.0], m[4:]])
        # sw = s0 * sum( ones*broadcast(rs) .* eq(s_ind, 1:numel(rs)), 2)
        sw = s0 * np.sum(rs[None, :] * (s_ind[:, None] == (np.arange(rs.size) + 1)[None, :]), axis=1)
    else:
        sw = s0

    mu2 = mu2_iso + mu2_aniso * (b_delta ** 2) * ((b_eta ** 2 + 3.0) / 3.0)

    # Numerically safe evaluation (similar spirit to MATLAB: s = real(s))
    MU2_EPS = 1e-30  # SI is small; keep extremely small epsilon
    mu2_safe = np.where(np.abs(mu2) < MU2_EPS, MU2_EPS, mu2)

    base = 1.0 + b * mu2_safe / d_iso
    base = np.maximum(base, 1e-30)
    exponent = -(d_iso ** 2) / mu2_safe

    s = sw * np.exp(exponent * np.log(base))
    return np.real(s)


# ----------------------------
# Options structure
# ----------------------------
@dataclass
class DTDGammaOpt:
    # bounds for [s0, d_iso, mu2_iso, mu2_aniso] (relative for s0; absolute for others in SI)
    fit_lb: np.ndarray  # length 4
    fit_ub: np.ndarray  # length 4

    fit_iters: int = 10
    guess_iters: int = 30

    do_multiple_s0: bool = True

    do_weight: bool = True
    weight_sthresh: float = 0.2
    weight_mdthresh: float = 1e-9
    weight_wthresh: float = 5.0

    do_pa_weight: bool = True

    # scipy least_squares options
    lsq_max_nfev: int = 200
    lsq_ftol: float = 1e-10
    lsq_xtol: float = 1e-10
    lsq_gtol: float = 1e-10
    lsq_method: str = "trf"  # supports bounds


# ----------------------------
# Helper weights (match MATLAB)
# ----------------------------
def weightfun(xps_b: np.ndarray, sthresh: float, mdthresh: float, wthresh: float) -> np.ndarray:
    """
    MATLAB:
      bthresh = -log(sthresh)/mdthresh
      weight = .5*(1-erf(wthresh*(b - bthresh)/bthresh));
    """
    bthresh = -np.log(sthresh) / mdthresh
    return 0.5 * (1.0 - erf(wthresh * (xps_b - bthresh) / bthresh))


def calc_weight_from_signal_samples(xps: Dict[str, Any]) -> np.ndarray:
    """
    MATLAB:
      if ~isfield(xps,'pa_w') w=ones
      else w = sqrt(pa_w/max(pa_w))
    """
    if "pa_w" not in xps or xps["pa_w"] is None:
        return np.ones_like(xps["b"], dtype=float)
    pa_w = np.asarray(xps["pa_w"], float)
    m = np.max(pa_w) if np.max(pa_w) > 0 else 1.0
    return np.sqrt(pa_w / m)


# ----------------------------
# Random guess generator (replacement for msf_fit_random_guess)
# ----------------------------
def random_guess_search(
    signal: np.ndarray,
    xps: Dict[str, Any],
    m_lbz: np.ndarray,
    m_ub: np.ndarray,
    weight: np.ndarray,
    ind: np.ndarray,
    ns: int,
    unit_to_SI: np.ndarray,
    guess_iters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample random m within bounds (in outside/SI units), evaluate residual, keep best.
    Returns m_guess (outside/SI units).
    """
    best_m = None
    best_r = np.inf

    # extend bounds for extra sw params
    lb = np.concatenate([m_lbz, 0.5 * np.ones(ns)])
    ub = np.concatenate([m_ub, 2.0 * np.ones(ns)])

    # ensure positivity for initial guesses where lb=0
    lb_pos = np.where(lb > 0, lb, 1e-30)

    for _ in range(guess_iters):
        # log-uniform for positive params to cover orders of magnitude reasonably
        m_try = np.empty_like(lb_pos)
        for k in range(m_try.size):
            if lb_pos[k] > 0 and ub[k] > 0:
                a, b = np.log(lb_pos[k]), np.log(ub[k])
                m_try[k] = np.exp(rng.uniform(a, b))
            else:
                m_try[k] = rng.uniform(lb[k], ub[k])

        # evaluate
        s_fit = dtd_gamma_fit2data(m_try, xps)
        r = np.sum(((signal - s_fit) * weight) [ind] ** 2)
        if r < best_r:
            best_r = r
            best_m = m_try

    return best_m


# ----------------------------
# Main fitter (Python port)
# ----------------------------
def dtd_gamma_1d_data2fit(
    signal: np.ndarray,
    xps: Dict[str, Any],
    opt: DTDGammaOpt,
    ind: Optional[np.ndarray] = None,
    rng_seed: int = 0,
) -> np.ndarray:
    """
    Port of dtd_gamma_1d_data2fit(signal, xps, opt, ind)

    Returns m in SI units (like MATLAB output):
      m = [s0, d_iso, mu2_iso, mu2_aniso, (optional sw...)]
    """
    signal = np.asarray(signal, float)
    n = signal.size

    if ind is None:
        ind = np.ones_like(signal, dtype=bool)
    else:
        ind = np.asarray(ind, dtype=bool)

    # Determine number of extra series scalings
    if ("s_ind" in xps) and opt.do_multiple_s0:
        s_ind = np.asarray(xps["s_ind"], int)
        ns = int(len(np.unique(s_ind[ind])) - 1)
        ns = max(ns, 0)
    else:
        ns = 0

    # unit_to_SI = [max(signal+eps) 1e-9 (1e-9)^2*[1 1] ones(1,ns)]
    smax = float(np.max(signal + np.finfo(float).eps))
    unit_to_SI = np.concatenate([[smax, 1e-9, (1e-9) ** 2, (1e-9) ** 2], np.ones(ns)])

    # bounds in SI/outside units
    m_lb = np.concatenate([np.array(opt.fit_lb, float), 0.5 * np.ones(ns)])
    m_ub = np.concatenate([np.array(opt.fit_ub, float), 2.0 * np.ones(ns)])

    # scale s0 bounds by max(signal)
    m_lb[0] *= smax
    m_ub[0] *= smax

    # avoid negative guess lower bound
    m_lbz = np.where(m_lb > 0, m_lb, 0.0)

    # transform bounds into "t-space" (local params)
    t_lb = m_lb / unit_to_SI
    t_ub = m_ub / unit_to_SI

    rng = np.random.default_rng(rng_seed)
    r_thr = np.inf
    m_keep = None

    b = np.asarray(xps["b"], float)
    pa_weight = calc_weight_from_signal_samples(xps) if opt.do_pa_weight else np.ones_like(b)

    for _ in range(opt.fit_iters):

        # initial weights
        weight = np.ones(n, dtype=float)
        if opt.do_weight:
            weight = weightfun(b, opt.weight_sthresh, opt.weight_mdthresh, opt.weight_wthresh)
        if opt.do_pa_weight:
            weight = weight * pa_weight

        # --- random guess in SI ---
        m_guess = random_guess_search(
            signal=signal,
            xps=xps,
            m_lbz=m_lbz[:4],           # only the first 4 are the model core bounds
            m_ub=m_ub[:4],
            weight=weight,
            ind=ind,
            ns=ns,
            unit_to_SI=unit_to_SI,
            guess_iters=opt.guess_iters,
            rng=rng,
        )
        t_guess = m_guess / unit_to_SI

        # residual function in t-space (includes weighting + ind)
        def residual(t: np.ndarray) -> np.ndarray:
            m = t * unit_to_SI
            s_fit = dtd_gamma_fit2data(m, xps)
            return (s_fit[ind] - signal[ind]) * weight[ind]

        # first fit
        res1 = least_squares(
            residual,
            x0=t_guess,
            bounds=(t_lb, t_ub),
            method=opt.lsq_method,
            max_nfev=opt.lsq_max_nfev,
            ftol=opt.lsq_ftol,
            xtol=opt.lsq_xtol,
            gtol=opt.lsq_gtol,
        )
        t = res1.x
        m = t * unit_to_SI

        # redo with updated MD in weightfun (m(2) in MATLAB)
        if opt.do_weight:
            md_est = float(m[1])
            weight2 = weightfun(b, opt.weight_sthresh, md_est, opt.weight_wthresh)
            if opt.do_pa_weight:
                weight2 = weight2 * pa_weight

            def residual2(t2: np.ndarray) -> np.ndarray:
                m2 = t2 * unit_to_SI
                s_fit2 = dtd_gamma_fit2data(m2, xps)
                return (s_fit2[ind] - signal[ind]) * weight2[ind]

            res2 = least_squares(
                residual2,
                x0=t,
                bounds=(t_lb, t_ub),
                method=opt.lsq_method,
                max_nfev=opt.lsq_max_nfev,
                ftol=opt.lsq_ftol,
                xtol=opt.lsq_xtol,
                gtol=opt.lsq_gtol,
            )
            t = res2.x
            m = t * unit_to_SI
            weight = weight2  # for residual check

        # check residual like MATLAB: sum(((signal-s_fit).*weight).^2)
        s_fit = dtd_gamma_fit2data(m, xps)
        res_val = np.sum(((signal - s_fit) * weight) ** 2)

        if res_val < r_thr:
            r_thr = res_val
            m_keep = m

    return m_keep


if __name__ == "__main__":
    # xps in SI units (match md-dmri convention)
    bvals_SI = np.asarray([0, 7, 10, 15, 20, 40, 50, 60, 100, 200, 400, 700, 1000, 1400, 2000]) * 1e-6  # s/mm^2 to s/m^2
    b_delta = None
    signal = np.array([1.0, 0.9, 0.85, 0.8, 0.75, 0.6, 0.55, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05])
    xps = {
        "b": bvals_SI,               # (N,)
        "b_delta": b_delta,          # (N,) optional but recommended
        "b_eta": np.zeros_like(bvals_SI),
        "s_ind": np.ones_like(bvals_SI, dtype=int),  # if multiple series
        # "pa_w": pa_w,              # optional
    }

    opt = DTDGammaOpt(
        fit_lb=np.array([0.2, 0.2e-9, -5e-18, -5e-18]),  # SI
        fit_ub=np.array([2.0, 3.0e-9,  5e-18,  5e-18]),
        fit_iters=10,
        guess_iters=50,
        do_weight=True,
        do_pa_weight=True,
    )

    m = dtd_gamma_1d_data2fit(signal, xps, opt)
    print("m =", m)  # [s0, d_iso, mu2_iso, mu2_aniso, ...]
