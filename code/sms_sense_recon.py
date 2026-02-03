# %%
import json
import os
from types import SimpleNamespace
import numpy as np
import scipy.io as sio
import mat73
from bart import bart
import plot_utils
import utils
import nibabel as nib
from pathlib import Path

# %% global parameters
with open("../config.json", "r") as f:
    cfg = json.load(f)

base = cfg["paths"]["base"]
typ = cfg["dataset"]["type"]
protocol = cfg["UIDnumber"]["stes"]
debug_level = cfg["debug"]["level"]

def render(rel_tmpl):
    return rel_tmpl.format(type=typ)

print(f"Processing {typ} data with protocol {protocol}...")
ps = SimpleNamespace() 
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
# bval = np.loadtxt(os.path.join("/data/users/cyang/RAWDATA_YANGCHENG/6_STEs_2mmiso_PA", "bval.bval"))
bval = np.loadtxt(os.path.join("/data/users/cyang/RAWDATA_YANGCHENG/" + protocol, "bval.bval"))
BVALS = bval
Nb = len(BVALS)


# %%
def save_nifti(img_data, ps, filename=None, ref_path=None, debug_level=0) -> None:
    out_path=ps.ip
    # affine = nib.load(ref_path).affine
    if debug_level >= 1:
        print(f"Reference path for NIfTI affine: {ref_path}")
    
    affine = nib.load(ref_path).affine 
    if debug_level >= 1:
        print(f"Affine matrix:\n{affine}")
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

    if debug_level >= 1:
        print(os.listdir(out_path))

# Nx, Ny, Nc, 
# print("Now getting kspace data...")
# data = np.zeros((164 * 2, 164, 16, 35, 67), dtype=np.complex128)
# %%
pref = protocol + "_" 
k_ngc_all = np.asarray(mat73.loadmat(os.path.join(ps.mat_p, pref+"k_ngc_all.mat"))["k_ngc_all"])  # (Nb, Nx, Ny, Nc, Nz)
k = np.transpose(k_ngc_all, (1,2,4,3,0))  # Nx, Ny, Nz, Nc, Nb
Nx, Ny, Nz, Nc, Nb = k.shape
Nz //= 2

mat_data = sio.loadmat(os.path.join(ps.mat_p, pref + "ngc_slice_grappa_data.mat"))
k_pparef = np.transpose(mat_data['k_pparef_ngc_reshape'][:, :, :, 33:35], (0,1,3,2))  # Nx, Ny, Nz, Nc
acs = utils.embed_center(k_pparef, np.zeros((Nx, Ny, Nz, Nc), dtype=np.complex128))
# %%
sens_low = bart(1, "ecalib -m1", acs[:,:,0:1,:])  # Nx, Ny, 1, Nc

# %%
pics_cmd = 'pics -S -l2 -r0.001 -i 10'

sense_prelim_all = np.zeros((Nx, Ny, Nz, Nb), dtype=np.complex128)
for i_slc in range(Nz):
    fft_bdelta_0 = k[:,:,i_slc:i_slc+1,:,:]  # Nx, Ny, 1, Nc, Nb
    sense_prelim = np.zeros((Nx, Ny, Nb), dtype=np.complex128)
    for i in range(Nb):
        sense_prelim[..., i] = bart(
            1, pics_cmd,
            fft_bdelta_0[..., i], sens_low
        )
    sense_prelim_all[:,:,i_slc,:] = sense_prelim

# %%
ref_path = os.path.join(ps.nii_p, "6_STEs_2mmiso_PA_4basis_bdelta0_bak_sense_prelim_abs.nii.gz")
save_nifti(sense_prelim_all, ps, filename=pref+"sense_prelim_all", ref_path=ref_path, debug_level=debug_level)