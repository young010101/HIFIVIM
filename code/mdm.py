import numpy as np
from pathlib import Path


class S:
    def __init__(self, nii_fn, xps):
        self.nii_fn = nii_fn
        self.xps = xps
    def __str__(self):
        return f"S(nii_fn={self.nii_fn}, xps=<{type(self.xps)}>)"


class XPS:
    """
    b_delta: float, STE 0, PTE (0, 1), LTE 1
    b: 10^9
    """
    def __init__(self, b, u, u_from_bvec, b_delta):
        self.b = b
        self.u = u
        self.u_from_bvec = u_from_bvec
        self.b_delta = b_delta
    def __str__(self):
        return f"XPS(b.shape={self.b.shape}, u.shape={self.u.shape}, b_delta.shape={self.b_delta.shape})"


def mdm_s_from_nii(nii_fn, b_delta) -> S:
    if not Path(nii_fn).is_file():
        raise ValueError("NIfTI file does not exist")
    xps = mdm_xps_from_nii_fn(nii_fn, b_delta)
    s = S(nii_fn, xps) 
    return s


def mdm_xps_from_nii_fn(nii_fn, b_delta) -> XPS:
    p = Path(nii_fn)
    if "".join(p.suffixes) != ".nii.gz":
        raise ValueError("Unsupported file extension")
    
    base = p.with_suffix('').with_suffix('')

    bval_fn = base.with_suffix('.bval')
    bvec_fn = base.with_suffix('.bvec')
    xps_mat = base.with_name(base.name + '_xps.mat')

    if Path(xps_mat).is_file():
        import scipy.io as sio
        xps_data = sio.loadmat(xps_mat)
        b = xps_data['b']  # in s/mm2
        u = xps_data['u']  # unit vectors
        u_from_bvec = xps_data.get('u_from_bvec', u)
        b_delta = xps_data['b_delta']
        xps = XPS(b, u, u_from_bvec, b_delta)
    elif Path(bval_fn).is_file() and Path(bvec_fn).is_file():
        b = np.loadtxt(bval_fn)
        u = np.loadtxt(bvec_fn).T
        N = len(b)
        b_delta = np.asarray(b_delta).repeat(N)
        xps = XPS(b, u, u, b_delta)
    else:
        raise ValueError("No bval/bvec or xps.mat file found")
    return xps    


def tm_tpars_to_1x6(b, b_delta, u):
    pass


def mdm_xps_from_bt(bt):
    pass

def mdm_nii_write(I, nii_fn, h):
    pass

def mdm_xps_save(xps, out_xps_fn): 
    pass
def mdm_s_powder_average(s, o_path, opt):
    pass

if __name__ == '__main__':
    xps = mdm_xps_from_nii_fn('/data/users/cyang/dtd_subspace/0119_ngc/DATA/brain/NII/6_STEs_2mmiso_PA_4basis_bdelta0_bak_sense_prelim_abs.nii.gz', b_delta=0)
    print(xps.b)
    print(xps.u)
    print(xps.b_delta)