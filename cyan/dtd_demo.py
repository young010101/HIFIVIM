import numpy as np

def dtd_forward(S0, Ds, ws, Bs):
    """
    DTD forward model (discrete mixture):
        S_i = S0 * sum_k w_k * exp( -tr(B_i D_k) )

    Args:
        S0: float
        Ds: (K,3,3) diffusion tensors (symmetric PSD recommended)
        ws: (K,) weights, nonnegative, sum to 1 recommended
        Bs: (N,3,3) b-tensors (symmetric PSD), e.g. b*g*g^T for LTE

    Returns:
        S: (N,) predicted signals
    """
    Ds = np.asarray(Ds, dtype=np.float64)   # (K,3,3)
    ws = np.asarray(ws, dtype=np.float64)   # (K,)
    Bs = np.asarray(Bs, dtype=np.float64)   # (N,3,3)

    # Compute M_{i,k} = tr(B_i D_k)
    # einsum: (N,3,3) (K,3,3) -> (N,K)
    tr_BD = np.einsum('nij,kji->nk', Bs, Ds)

    # A_{i,k} = exp(-tr(B_i D_k))
    A = np.exp(-tr_BD)

    # S = S0 * A @ w
    S = float(S0) * (A @ ws)
    return S


def make_B_lte(bvals, bvecs):
    """
    Build b-tensors for linear tensor encoding (LTE):
        B_i = b_i * g_i g_i^T

    Args:
        bvals: (N,)
        bvecs: (N,3) (unit vectors recommended)

    Returns:
        Bs: (N,3,3)
    """
    bvals = np.asarray(bvals, dtype=np.float64)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    g = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)

    # outer: (N,3,1)*(N,1,3) -> (N,3,3)
    ggT = g[:, :, None] * g[:, None, :]
    Bs = bvals[:, None, None] * ggT
    return Bs


# ---- tiny demo ----
if __name__ == "__main__":
    S0 = 1000.0

    # Two-component DTD (K=2)
    D1 = np.diag([1.7e-3, 0.4e-3, 0.4e-3])  # prolate-ish
    D2 = np.diag([0.9e-3, 0.9e-3, 0.9e-3])  # isotropic
    Ds = np.stack([D1, D2], axis=0)
    ws = np.array([0.6, 0.4])

    # Measurements (N=6)
    bvecs = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
    ], dtype=np.float64)
    bvals = np.full((bvecs.shape[0],), 1000.0)

    Bs = make_B_lte(bvals, bvecs)
    S = dtd_forward(S0, Ds, ws, Bs)
    print(S)
