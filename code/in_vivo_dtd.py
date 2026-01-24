# %% import region
import random
import time
from types import SimpleNamespace
# import importlib
import cyan_utils
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import matplotlib.pyplot as plt
from utils import get_composite_sens
from bart import bart
import utils
import dict_gen_dtd
import plot_utils
import nibabel as nib
import os
import Phantom_utils
from scipy.io import loadmat
import h5py
import cfl
import json
from tqdm import tqdm
import multiprocessing
import tempfile
import shutil

# %% global parameters
with open("../config.json", "r") as f:
    cfg = json.load(f)

base = cfg["paths"]["base"]
typ = cfg["dataset"]["type"]
protocol = cfg["dataset"]["protocol"]
debug_level = cfg["debug"]["level"]

def render(rel_tmpl):
    return rel_tmpl.format(type=typ, protocol=protocol)

print(f"Processing {typ} data with protocol {protocol}...")
# %%
args = SimpleNamespace()
args.outdir = '../Phantom/'
ps = SimpleNamespace() 
PROTOCOL = protocol
ps.nii_p = os.path.join(base, render(cfg["layout"]["input_rel"]))  # <- actual input data
ps.mat_p =  os.path.join(base, render(cfg["layout"]["mat_rel"]))  # <- .mat files path
ps.bart_p = os.path.join(base, render(cfg["layout"]["bart_rel"]))  # <- .bart files path
ps.ip = ps.nii_p
ps.op = os.path.join(base, render(cfg["layout"]["output_rel"]))  # <- store output here
ps.zp = os.path.join(base, cfg["paths"]["tmp_rel"])  # <- store temporary files here
if debug_level >= 1:
    print(ps.nii_p)
    print(ps.mat_p)
    print(ps.bart_p)
POINTS = [(82, 82), (50, 50), (30, 130), (100, 60), (45, 90)]
bval = np.loadtxt(os.path.join("/data/users/cyang/RAWDATA_YANGCHENG/6_STEs_2mmiso_PA", "bval.bval"))
BVALS = bval
num_bvals = len(BVALS)

# %% define some useful functions


def simulate_coil_ksp(img_xyb, num_coils=16, device=sp.cpu_device):
    """
    Parameters
    ----------
    img_xyb :   Nx, Ny, Nb

    Returns
    ------
    ksp :       Nc, Nx, Ny, Nb 
    mps_cxy :   Nc, Nx, Ny
    """
    img_shape = img_xyb.shape[:2]

    with sp.Device(device):
        mps_cxy = mr.birdcage_maps((num_coils, *img_shape))
        img = sp.to_device(img_xyb, device)
        coil_imgs = img[None, ...] * mps_cxy[..., None]
        ksp = sp.fft(coil_imgs, axes=(1, 2))

    return ksp, mps_cxy


def save_nifti(img_data, ps, filename=None, ref_path=None, debug=False) -> None:
    out_path=ps.op
    # affine = nib.load(ref_path).affine
    ref_path = os.path.join(args.outdir, 'Phantom_T1.nii.gz') if ref_path is None else ref_path
    
    affine = nib.load(ref_path).affine 
    img = nib.Nifti1Image(img_data, affine)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    nib.save(img, os.path.join(out_path, f'{filename}.nii.gz'))
    # save bvals
    np.savetxt(os.path.join(out_path, f'{filename}.bval'), BVALS.reshape(1, len(BVALS)), fmt='%d', delimiter=' ')
    # save unit bvecs
    bvecs = np.zeros((3, len(BVALS)))
    bvecs[0, :] = 1  # x direction
    np.savetxt(os.path.join(out_path, f'{filename}.bvec'), bvecs, fmt='%d', delimiter=' ')

    if debug:
        print(os.listdir(out_path))

def apply_effects(img_xyb, add_phase=False, add_noise=False, noise_sigma=20, show=False):
    out = np.asarray(img_xyb, dtype=np.complex128).copy()
    if add_phase:
        out = Phantom_utils.add_phase(args, out, show=show)
    if add_noise:
        out = utils.add_noise(out, noise_sigma, return_img=False)
    return out


def build_variants(img_xyb, noise_sigma=20, show=False):
    variant_flags = [
        ("clean", False, False),
        ("phase", True, False),
        ("noise", False, True),
        ("phase_noise", True, True),
    ]
    variants = {}
    for name, add_phase, add_noise in variant_flags:
        variants[name] = apply_effects(
            img_xyb, add_phase=add_phase, add_noise=add_noise,
            noise_sigma=noise_sigma, show=show
        )
    return variants


def run_pipeline(composite_dtd, basis, num_bvals):
    ksp_bdelta_0, _ = simulate_coil_ksp(composite_dtd)
    fft_bdelta_0 = ksp_bdelta_0.transpose(1, 2, 0, 3)[:, :, None, :, :]

    ksp_calib = np.mean(ksp_bdelta_0, axis=-1)
    app = mr.app.EspiritCalib(ksp_calib, calib_width=24, device=sp.Device(0))
    mps_estimated = app.run()
    mps_estimated = sp.to_device(mps_estimated, sp.cpu_device)
    sens_maps_expand = np.moveaxis(mps_estimated, 0, -1)[..., None, :]

    sense_prelim = np.zeros((*composite_dtd.shape[:2], num_bvals), dtype=np.complex128)
    for i in range(num_bvals):
        sense_prelim[..., i] = bart(
            1, 'pics -S -l2 -r0.001 -i 10',
            fft_bdelta_0[..., i], sens_maps_expand
        )

    _composite_sens, _ = get_composite_sens(
        sense_prelim, sens_maps_expand, visualize="True"
    )
    composite_sens = np.expand_dims(
        np.transpose(_composite_sens, (0, 1, 4, 2, 3)), axis=4
    )

    fft_bdelta_0_expand = np.expand_dims(fft_bdelta_0, axis=4)
    basis2 = basis[..., :2]
    _, recon_fmac_basis = utils.llr_recon_with_retry(
        fft_bdelta_0_expand,
        composite_sens,
        basis2,
        use_basis=True,
        lambda1=0.001,
        lambda2=0.001,
    )
    return recon_fmac_basis


def show_variant_grid(results, title_prefix="Recon |mean|"):
    plt.figure(figsize=(8, 8))
    for idx, (name, recon_fmac_basis) in enumerate(results.items()):
        img = np.mean(np.abs(recon_fmac_basis.squeeze()), axis=-1)
        ax = plt.subplot(2, 2, idx + 1)
        ax.imshow(img, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    plt.suptitle(title_prefix)


def load_cell_array(path, key):
    with h5py.File(path, "r") as f:
        refs = np.array(f[key])          # convert to ndarray of object refs
        out = []
        for ref in refs.ravel():         # flatten to 1D
            ds = f[ref]
            arr = ds[()]                 # read dataset
            if arr.dtype.names is not None and set(arr.dtype.names) == {"real", "imag"}:
                arr = arr["real"] + 1j * arr["imag"]
            out.append(np.array(arr))
    # Try to stack into proper numeric array
    try:
        return np.stack(out).reshape(refs.shape + out[0].shape)
    except:
        return np.array(out, dtype=object)


def bart_retry(nargout, cmd, *args, retries=3, base_sleep=0.5, jitter=0.2, tmp_root=None):
    """
    Retry wrapper for python-bart `bart(...)`.
    - retries: number of extra attempts (total attempts = retries+1)
    - tmp_root: if set, create a per-attempt TMPDIR under this root to avoid collisions
    """
    last_err = None

    for attempt in range(retries + 1):
        # optional per-attempt temp dir isolation (helps a lot in parallel)
        tmpdir = None
        old_tmp = os.environ.get("TMPDIR", None)

        try:
            if tmp_root is not None:
                tmpdir = tempfile.mkdtemp(prefix="bart_", dir=tmp_root)
                os.environ["TMPDIR"] = tmpdir

            return bart(nargout, cmd, *args)

        except Exception as e:
            last_err = e

            # cleanup temp dir
            if tmpdir is not None:
                shutil.rmtree(tmpdir, ignore_errors=True)
                if old_tmp is None:
                    os.environ.pop("TMPDIR", None)
                else:
                    os.environ["TMPDIR"] = old_tmp

            if attempt == retries:
                break

            sleep_s = base_sleep * (2 ** attempt) * (1.0 + random.uniform(-jitter, jitter))
            time.sleep(max(0.0, sleep_s))

    raise last_err

def in_vivo_pipe(k, acs, num_coils=None):

    sens_fft_bdelta_0_by_bart = get_sens_by(k, acs)
    
    if debug_level >= 1:
        print('Sens map shape by bart ecalib:', sens_fft_bdelta_0_by_bart.shape)
        plot_utils.help_show_imgs(sens_fft_bdelta_0_by_bart.squeeze())

    _tmp =bart_retry( 1, 'pics -S -l2 -r0.001 -i 10', k, sens_fft_bdelta_0_by_bart)
    if debug_level >= 1:
        print('Tmp recon shape:', _tmp.shape)
        plot_utils.help_show_imgs(np.abs(_tmp[...,None]), cmap='gray')
    
    return _tmp

def get_sens_by(k, acs):

    mbref_embed = utils.embed_center(acs, k)
    sens_fft_bdelta_0_by_bart = bart_retry(1, "ecalib -m1", mbref_embed)

    return sens_fft_bdelta_0_by_bart
# %%
k_ngc_all = load_cell_array(os.path.join(ps.mat_p, "k_ngc_all.mat"), "k_ngc_all")
k = np.transpose(k_ngc_all[:,0,0,...].squeeze(), (3, 2, 1, 0))[:,:,None,:,:]

mat_data = loadmat(os.path.join(ps.mat_p, "ngc_slice_grappa_data.mat"))
k_pparef = mat_data['k_pparef_ngc_reshape'][:, :, :, 33][:,:,None,:]

num_x, num_y, num_z, num_c, num_b = k.shape
recon_demo = np.zeros((num_x, num_y, num_b), dtype=np.complex128)
# for b in tqdm(range(10,num_b)):
#     recon_demo[..., b] = in_vivo_pipe(k[..., b], k_pparef, num_coils=16) 

plot_utils.help_show_imgs(np.abs(recon_demo[...,:16]), cmap='gray')

# with multiprocessing.Pool() as pool:
#     results = pool.starmap(in_vivo_pipe, [(k[..., b], k_pparef) for b in range(num_b)])
#     for b, result in enumerate(results):
#         recon_demo[..., b] = result

# %%
fft_bdelta_0 = k
sens_maps_expand = get_sens_by(fft_bdelta_0[..., 0], k_pparef)

basis = dict_gen_dtd.basis_pipeline(bvals=BVALS, num_basis=5, debug=True)

sense_prelim = np.zeros((num_x, num_y, num_b), dtype=np.complex128)
for i in range(num_bvals):
    sense_prelim[..., i] = bart(
        1, 'pics -S -l2 -r0.001 -i 10',
        fft_bdelta_0[..., i], sens_maps_expand
    )

_composite_sens, _ = utils.get_composite_sens(
    sense_prelim, sens_maps_expand, bvals=BVALS, visualize="True"
)
composite_sens = np.expand_dims(
    np.transpose(_composite_sens, (0, 1, 4, 2, 3)), axis=4
)

fft_bdelta_0_expand = np.expand_dims(fft_bdelta_0, axis=4)
basis2 = basis[..., :2]
_, recon_fmac_basis = utils.llr_recon_with_retry(
    fft_bdelta_0_expand,
    composite_sens,
    basis2,
    use_basis=True,
    lambda1=0.001,
    lambda2=0.001,
)
# Ensure output directory exists before writing CFL
os.makedirs(ps.bart_p, exist_ok=True)
cfl.writecfl(os.path.join(ps.bart_p, 'phan_dtd_recon_2basis'), recon_fmac_basis)