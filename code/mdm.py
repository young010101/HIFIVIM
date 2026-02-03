from pathlib import Path


class S:
    def __init__(self, nii_fn, xps):
        self.nii_fn = nii_fn
        self.xps = xps


class XPS:
    """
    b_delta: float, STE 0, PTE (0, 1), LTE 1
    """
    def __init__(self, bval_fn, bvec_fn, b_delta):
        self.bval_fn = bval_fn
        self.bvec_fn = bvec_fn
        self.b_delta = b_delta


def mdm_s_from_nii(nii_fn, b_delta) -> S:
    xps = mdm_xps_from_nii_fn(nii_fn, b_delta)
    s = S(nii_fn, xps) 
    return s


def mdm_xps_from_nii_fn(nii_fn, b_delta) -> XPS:
    p = Path(nii_fn)
    ext = "".join(p.suffixes)

    if ext == ".nii.gz":
        file = p.name[: -len(ext)] 
    else:
        raise ValueError("Unsupported file extension")

    bval_fn = str(file + '.bval')
    bvec_fn = str(file + '.bvec')
    xps = XPS(bval_fn, bvec_fn, b_delta)
    return xps    


def tm_tpars_to_1x6(b, b_delta, u):
    pass


def mdm_xps_from_bt(bt):
    pass


if __name__ == '__main__':
    xps = mdm_xps_from_nii_fn('/data/users/cyang/RAWDATA_YANGCHENG/6_STEs_2mmiso_PA/_STEs_2mmiso_PA_20260118122029_601_slc34.nii.gz', b_delta=0)
    print(xps.bval_fn)
    print(xps.bvec_fn)
    print(xps.b_delta)