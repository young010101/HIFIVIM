# %% import region
from types import SimpleNamespace
# import importlib
import cyan_utils
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
from utils import BVALS, get_composite_sens
from bart import bart
import utils
import dict_gen_dtd
import plot_utils
import nibabel as nib
import os
import Phantom_utils
from scipy.io import loadmat

# %% global parameters
args = SimpleNamespace()
args.outdir = '../Phantom/'
ps = SimpleNamespace() 
ps.bp = '/data/users/cyang/dtd_subspace/0119_ngc'  # <- base path
PROTOCOL = "6_STEs_2mmiso_PA"
ps.ip = os.path.join(ps.bp, 'DATA', 'brain', PROTOCOL, 'NII')  # <- actual input data
ps.mat_p =  os.path.join(ps.bp, 'DATA', 'brain', PROTOCOL, 'MAT')
ps.op = os.path.join(ps.bp, 'processed', 'brain', PROTOCOL)  # <- store output here
ps.zp = os.path.join(ps.bp, 'tmp')  # <- store temporary files here

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


def help_show_imgs(imgs, cmap='gray'):
    utils.show_imgs(np.abs(imgs.transpose(2,0,1)), cmap=cmap)


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


# %%

basis = dict_gen_dtd.basis_pipeline(bvals=BVALS, num_basis=5, debug=True)

# Load .mat file and extract variables
mat_data = loadmat(os.path.join(ps.mat_p, "ngc_slice_grappa_data.mat"))
import h5py
with h5py.File(os.path.join(ps.mat_p, "k_ngc_all.mat"), 'r') as f:
    k_ngc_all = f['k_ngc_all'][:]
import h5py
import numpy as np
import os

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

k_ngc_all = load_cell_array(os.path.join(ps.mat_p, "k_ngc_all.mat"), "k_ngc_all")
k_ngc_all = k_ngc_all.squeeze()
Img_Grappa_all = mat_data['Img_Grappa_all']
k_pparef_ngc = mat_data['k_pparef_ngc']

# k_ngc_all: (57, 4, 64, 112, 112)
# take first slice of the 2nd dim (index 0), then reorder to (64, 112, 112, 57)
k_first = k_ngc_all[:, 0, :, :, :]          # (57, 64, 112, 112)
k_reordered = np.transpose(k_first, (1, 2, 3, 0))  # (64, 112, 112, 57)

ksp_bdelta_0 = k_reordered

_fft_bdelta_0 = ksp_bdelta_0.transpose(1, 2, 0, 3)[:, :, None, :, :]
print(_fft_bdelta_0.shape)
num_coils = 16

# Apply coil compression per last dimension (b-value)
fft_bdelta_0 = np.zeros(
    (*_fft_bdelta_0.shape[:3], num_coils, _fft_bdelta_0.shape[-1]),
    dtype=_fft_bdelta_0.dtype,
)
for b_idx in range(_fft_bdelta_0.shape[-1]):
    fft_bdelta_0[..., b_idx] = bart(1, f"cc -p{num_coils} -A -S", _fft_bdelta_0[..., b_idx])

ksp_bdelta_0 = fft_bdelta_0.squeeze().transpose(2, 0, 1, 3)[:, :, :, :]

# for b_idx in range(_fft_bdelta_0.shape[-1]):
b_idx = 0
sens_fft_bdelta_0_by_bart = bart(1, "ecalib -m1", fft_bdelta_0[..., b_idx])
print('Sens map shape by bart ecalib:', sens_fft_bdelta_0_by_bart.shape)
help_show_imgs(sens_fft_bdelta_0_by_bart.squeeze())

ksp_calib = np.mean(ksp_bdelta_0, axis=-1)
app = mr.app.EspiritCalib(ksp_calib, calib_width=24, device=sp.Device(0))
mps_estimated = app.run()
mps_estimated = sp.to_device(mps_estimated, sp.cpu_device)
sens_maps_expand = np.moveaxis(mps_estimated, 0, -1)[..., None, :]

sense_prelim = np.zeros((*fft_bdelta_0.shape[:2], num_bvals), dtype=np.complex128)
for i in range(num_bvals):
    sense_prelim[..., i] = bart(
        1, 'pics -S -l2 -r0.001 -i 10',
        fft_bdelta_0[..., i], sens_maps_expand
    )
print('Sense prelim shape:', sense_prelim.shape)

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
import cfl
# Ensure output directory exists before writing CFL
os.makedirs(ps.op, exist_ok=True)
cfl.writecfl(os.path.join(ps.op, 'phan_dtd_recon_2basis'), recon_fmac_basis)

# %% generate phantom
mean_diff, var_iso, var_aniso, _dtd_gamma_bdelta_0, _dtd_gamma_bdelta_1, masks = cyan_utils.make_phantom(args, show=True, points=POINTS)
if False:
    dtd_gamma_bdelta_0 = np.rot90(_dtd_gamma_bdelta_0, k=1)
else:
    dtd_gamma_bdelta_0 = _dtd_gamma_bdelta_0
if False:
    cyan_utils.show_15_bvals(dtd_gamma_bdelta_0)

basis = dict_gen_dtd.basis_pipeline(bvals=BVALS, num_basis=5, debug=True)
print('Basis shape:', basis.shape)  # Nb, num_basis

base_dtd = np.asarray(dtd_gamma_bdelta_0, dtype=np.complex128).copy()
variants = build_variants(base_dtd, noise_sigma=20, show=False)

recon_results = {}
for name, composite_dtd in variants.items():
    recon_results[name] = run_pipeline(composite_dtd, basis=basis, num_bvals=num_bvals)

show_variant_grid(recon_results, title_prefix="Recon |mean| across bvals")

# %% gen dict
ref_recon = recon_results["clean"]
help_show_imgs(ref_recon.squeeze())
help_show_imgs(dtd_gamma_bdelta_0)
help_show_imgs(np.abs(ref_recon.squeeze()) - np.abs(dtd_gamma_bdelta_0), cmap='bwr')
help_show_imgs(ref_recon.squeeze() - dtd_gamma_bdelta_0)
print('MSE:', np.mean((np.abs(ref_recon.squeeze()) - dtd_gamma_bdelta_0)**2))
plot_utils.plot_recon_vs_ivim(
    (
        ref_recon.squeeze().real, 
        recon_results['noise'].squeeze().real, 
        dtd_gamma_bdelta_0), 
    BVALS, POINTS, recon_label="2 basis", ivim_scale=1.0)

plot_utils.plot_recon_vs_ivim(
    {
        "clean": ref_recon.squeeze().real, 
        "noise": recon_results['noise'].squeeze().real, 
        "phase": recon_results['phase'].squeeze().real,
        "phase_noise": recon_results['phase_noise'].squeeze().real,
        "ground_truth": dtd_gamma_bdelta_0
    }, 
    BVALS, POINTS, recon_label="2 basis", ivim_scale=1.0,
    figure_kwargs={'figsize': (15, 10)})

plot_utils.plot_recon_vs_ivim(
    {
        "clean": ref_recon.squeeze().real, 
        "phase": recon_results['phase'].squeeze().real,
    }, 
    BVALS, POINTS, recon_label="2 basis", ivim_scale=1.0,
    figure_kwargs={'figsize': (15, 10)})



# %%
save_nifti(ref_recon.squeeze(), ps=ps, filename='phan_dtd_recon_2basis', ref_path=None, debug=True)
