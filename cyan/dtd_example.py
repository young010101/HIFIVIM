import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utilities: sphere sampling, tensors, Voigt(6) with sqrt(2) shear scaling
# -----------------------------
SQRT2 = np.sqrt(2.0)


def fibonacci_sphere(n: int) -> np.ndarray:
    """
    Approximate uvec_elstat(n, ...) using Fibonacci sphere sampling.
    Returns (n,3) unit vectors roughly uniformly distributed on S^2.
    """
    # Golden angle
    ga = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = ga * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    u = np.stack([x, y, z], axis=1)
    # normalize (numerical safety)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def tensor_from_ad_rd(ad: float, rd: float, n: np.ndarray) -> np.ndarray:
    """
    Construct 3x3 diffusion tensor with axial diffusivity ad and radial rd,
    oriented along unit vector n.
    D = rd*I + (ad-rd) * (n n^T)
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    I = np.eye(3)
    return rd * I + (ad - rd) * np.outer(n, n)


def tensor_to_voigt6(D: np.ndarray) -> np.ndarray:
    """
    Voigt-like 6-vector with sqrt(2) scaling of off-diagonal terms,
    consistent with many diffusion tensor toolboxes:
      [Dxx, Dyy, Dzz, sqrt(2)*Dxy, sqrt(2)*Dxz, sqrt(2)*Dyz]
    This makes dot(voigt(D), voigt(B)) match Frobenius inner product D:B.
    """
    return np.array([
        D[0, 0],
        D[1, 1],
        D[2, 2],
        SQRT2 * D[0, 1],
        SQRT2 * D[0, 2],
        SQRT2 * D[1, 2],
    ], dtype=float)


def batch_tensors_to_voigt6(Ds: np.ndarray) -> np.ndarray:
    """
    Ds: (M,3,3) -> (M,6)
    """
    M = Ds.shape[0]
    out = np.empty((M, 6), dtype=float)
    out[:, 0] = Ds[:, 0, 0]
    out[:, 1] = Ds[:, 1, 1]
    out[:, 2] = Ds[:, 2, 2]
    out[:, 3] = SQRT2 * Ds[:, 0, 1]
    out[:, 4] = SQRT2 * Ds[:, 0, 2]
    out[:, 5] = SQRT2 * Ds[:, 1, 2]
    return out


def b_tensor_axial_radial_to_voigt6(b_axial: float, b_radial: float, u: np.ndarray) -> np.ndarray:
    """
    Make an axisymmetric b-tensor with principal direction u,
    with eigenvalues:
      along u:      b_axial
      orthogonal:   b_radial (degenerate, multiplicity 2)

    B = b_radial*I + (b_axial - b_radial) * (u u^T)
    Return Voigt6 representation with sqrt(2) shear scaling.
    """
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    I = np.eye(3)
    B = b_radial * I + (b_axial - b_radial) * np.outer(u, u)
    return tensor_to_voigt6(B)


def merge_xps(x_list):
    """
    Minimal replacement for mdm_xps_merge(x) in the example:
    concatenate fields into one dict and create s_ind (series index).
    """
    b_all, bt_all, bdel_all, beta_all, s_ind_all = [], [], [], [], []
    for idx, x in enumerate(x_list, start=1):
        n = x["n"]
        b_all.append(x["b"])
        bt_all.append(x["bt"])
        bdel_all.append(x["b_delta"])
        beta_all.append(x["b_eta"])
        s_ind_all.append(np.full(n, idx, dtype=int))
    xps = {
        "n": sum(xx["n"] for xx in x_list),
        "b": np.concatenate(b_all, axis=0),
        "bt": np.vstack(bt_all),
        "b_delta": np.concatenate(bdel_all, axis=0),
        "b_eta": np.concatenate(beta_all, axis=0),
        "s_ind": np.concatenate(s_ind_all, axis=0),
    }
    return xps


# -----------------------------
# Optional: simple "fit" in log-domain (2nd order in b) per b_delta group
# This is NOT a 1:1 port of dtd_pa_1d_data2fit, but a demonstration.
# -----------------------------
def fit_log_quadratic_per_group(S: np.ndarray, xps: dict) -> np.ndarray:
    """
    For each group (b_delta series), fit:
      log S = c0 + c1*b + c2*b^2
    using ordinary least squares.
    Return S_fit for all points.
    """
    S_fit = np.zeros_like(S)
    for g in np.unique(xps["s_ind"]):
        ind = xps["s_ind"] == g
        b = xps["b"][ind]
        y = np.log(np.maximum(S[ind], 1e-300))
        A = np.stack([np.ones_like(b), b, b**2], axis=1)
        # Solve min ||A c - y||^2
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ c
        S_fit[ind] = np.exp(yhat)
    return S_fit


# -----------------------------
# Main example
# -----------------------------
def dtd_example():
    # ---- generate three types of diffusion tensors (DTD samples)
    ad_1, rd_1, n_1 = 2.9e-9, 0.5e-9, 400
    ad_2, rd_2, n_2 = 2.9e-9, 1.5e-9, 300
    d_iso, n_3 = 3.0e-9, 100

    u1 = fibonacci_sphere(n_1)
    u2 = fibonacci_sphere(n_2)
    u3 = fibonacci_sphere(n_3)

    D1 = np.stack([tensor_from_ad_rd(ad_1, rd_1, u) for u in u1], axis=0)
    D2 = np.stack([tensor_from_ad_rd(ad_2, rd_2, u) for u in u2], axis=0)
    D3 = np.stack([tensor_from_ad_rd(d_iso, d_iso, u) for u in u3], axis=0)

    Ds = np.concatenate([D1, D2, D3], axis=0)       # (M,3,3)
    dt = batch_tensors_to_voigt6(Ds)                # (M,6)  Voigt notation

    # ---- setup experiment (powder-averaged: one direction sufficient)
    u = np.array([1.0, 0.0, 0.0])
    b = np.linspace(np.finfo(float).eps, 3e9, 10)   # (10,)
    b_delta_list = [0.0, 0.7, 1.0]

    x_list = []
    for b_delta in b_delta_list:
        # axisymmetric btensor parameterization
        b_radial = b * (1.0 - b_delta) / 3.0
        b_axial = b - 2.0 * b_radial

        bt = np.stack([b_tensor_axial_radial_to_voigt6(ba, br, u) for ba, br in zip(b_axial, b_radial)], axis=0)
        x_list.append({
            "n": b.size,
            "b": b.copy(),
            "bt": bt,  # (n,6)
            "b_delta": np.full_like(b, b_delta, dtype=float),
            "b_eta": np.zeros_like(b, dtype=float),
        })

    xps = merge_xps(x_list)

    # ---- synthetic measurement: S_i = mean_k exp(- D_k : B_i)
    # In Voigt form with sqrt(2) scaling, D:B == dot(dt_row, bt_row)
    # dt (M,6), bt (N,6) => dt @ bt.T => (M,N)
    expo = - dt @ xps["bt"].T
    S = np.mean(np.exp(expo), axis=0)  # (N,)

    # ---- fit a model and generate fitted signal (placeholder)
    # MATLAB chooses c_model=3 (dtd_pa). Here we provide a simple quadratic log-fit.
    S_fit = fit_log_quadratic_per_group(S, xps)

    # ---- plotting
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)

    l_str = []
    for g in np.unique(xps["s_ind"]):
        ind = xps["s_ind"] == g
        ax1.semilogy(xps["b"][ind] * 1e-9, S[ind], "o", linewidth=2)
        ax1.semilogy(xps["b"][ind] * 1e-9, S_fit[ind], "-", linewidth=2)
        l_str.append(f"b_Δ = {np.mean(xps['b_delta'][ind]):.2f}")

    ax1.set_xlabel("b [um^2/ms]")
    ax1.set_ylabel("Signal")
    ax1.legend(l_str, frameon=False)
    ax1.set_title("Synthetic powder-averaged signal (DTD forward)")

    # ---- right panel: visualize tensors (simple proxy for mplot_tensors_in_voxel)
    # We'll plot scatter of (MD, FA) for the DTD samples as an interpretable summary.
    # (Not the same as ellipsoid rendering, but conveys the distribution)
    ax2 = fig.add_subplot(1, 2, 2)

    # compute MD and FA of each sample tensor
    evals = np.linalg.eigvalsh(Ds)  # (M,3)
    md = np.mean(evals, axis=1)
    # FA formula
    num = np.sqrt(1.5) * np.linalg.norm(evals - md[:, None], axis=1)
    den = np.sqrt(np.sum(evals**2, axis=1))
    fa = np.where(den > 0, num / den, 0.0)

    ax2.plot(md * 1e9, fa, ".", markersize=3)  # scale to (um^2/ms)
    ax2.set_xlabel("MD [um^2/ms]")
    ax2.set_ylabel("FA")
    ax2.set_title("DTD samples summary (MD vs FA)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dtd_example()
