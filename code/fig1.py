# %%
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from pathlib import Path
import mdm
import glob
import matplotlib.gridspec as gridspec
# %% ======================compare reconstructions======================
p = Path("/data/users/cyang/dtd_subspace/0119_ngc/DATA/brain_2mmiso/NII")
names_dict = {
    "dicom": "_STEs_2mmiso_PA_20260118122029_601_slc37_36_rot180_sorted_pa.nii.gz",
    "grappa": "6_STEs_2mmiso_PA_grappa_gold_pa.nii.gz",
    "sense": "6_STEs_2mmiso_PA_sense_prelim_all_abs_pa.nii.gz"
}
names = names_dict.values()
n_cols = len(names)

ss = [mdm.mdm_s_from_nii(p / name, 0) for name in names]
imgs = [nib.load(s.nii_fn).get_fdata() for s in ss]

# b_s_mm2 = [1000, 2000]
b_s_mm2 = ss[0].xps.b / 10**6
Nb = len(b_s_mm2)

font_style = {
    "fontsize": 16,
    "fontweight": "bold",
}
idx_slc = 0

# fig, axes = plt.subplots(Nb, n_cols, figsize=(6*n_cols, 6*Nb), gridspec_kw={"hspace": 0, "wspace": 0})
fig, axes = plt.subplots(Nb, n_cols, figsize=(6*n_cols, 6*Nb))

for i in range(n_cols):
    for j in range(Nb):
        ax = axes[j, i]
        idx = np.where(np.isclose(ss[i].xps.b, b_s_mm2[j] * 10**6))[0][0]
        im = ax.imshow(np.rot90(imgs[i][:, :, idx_slc, idx], k=-1), cmap="gray")
        plt.colorbar(im)
        if j == 0:
            ax.set_title(f"{list(names_dict.keys())[i]}", pad=12, **font_style)
        if i == 0:
            ax.set_ylabel(f"b = {b_s_mm2[j]:.0f} s/mm$^2$", **font_style)

# axes[0, 0].set_ylabel("Magnitude ($I_0$)", **font_style)

for ax in axes.flatten():
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"../figs/{''.join(names_dict.keys())}.png", dpi=300)
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
from pathlib import Path

p = Path("/data/users/cyang/dtd_subspace/0119_ngc/DATA/brain_2mmiso/NII")

names_dict = {
    "Conventional": "_STEs_2mmiso_PA_20260118122029_601_slc37_36_rot180_sorted_pa.nii.gz",
    "Proposed": "6_STEs_2mmiso_PA_sense_prelim_all_abs_pa.nii.gz"
}
names = list(names_dict.values())
keys = list(names_dict.keys())
n_cols = len(names)

# ������������������������ mdm
ss = [mdm.mdm_s_from_nii(p / name, 0) for name in names]
imgs = [nib.load(s.nii_fn).get_fdata() for s in ss]

# �������������� b ��������3������������
unique_b = np.unique(np.round(ss[0].xps.b / 10**6))
if len(unique_b) >= 3:
    b_s_mm2 = [unique_b[0], unique_b[len(unique_b)//2], unique_b[-1]]
else:
    b_s_mm2 = unique_b

Nb = len(b_s_mm2)

# --- ���� 1: �������� ---
font_style = {
    "fontsize": 24,  # �� 16 �������� 24���������������������� 28 �� 32
    "fontweight": "bold",
}
idx_slc = 0

fig = plt.figure(figsize=(6 * n_cols + 1, 6 * Nb))
width_ratios = [1] * n_cols + [0.05, 0.1]
gs = gridspec.GridSpec(Nb, n_cols + 2, width_ratios=width_ratios, wspace=0.0, hspace=0.0)

for j in range(Nb):
    im = None
    for i in range(n_cols):
        ax = fig.add_subplot(gs[j, i])
        
        idx = np.where(np.isclose(ss[i].xps.b / 10**6, b_s_mm2[j], atol=0.1))[0][0]
        
        img_slice = np.rot90(imgs[i][:, :, idx_slc, idx], k=-1)
        im = ax.imshow(img_slice, cmap="gray")
        
        if j == 0:
            ax.set_title(keys[i], pad=16, **font_style) # pad ����������������������������������������
        if i == 0:
            ax.set_ylabel(f"b = {b_s_mm2[j]:.0f} s/mm$^2$", **font_style)
            
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        cax = fig.add_subplot(gs[j, -1])
        cbar = fig.colorbar(im, cax=cax)
        # --- ���� 2: ���� colorbar ������ ---
        cbar.set_ticks([]) 

save_path = f"../figs/Conventional_Proposed_bvals.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {save_path}")
plt.show()
# %% =====================signal decay w/o LTE======================
points_dict = {"csf": (46, 52), "wm": (40, 40), "gm": (30, 35)}
points = list(points_dict.values())
num_points = len(points)
fig, axes = plt.subplots(1, num_points+1, figsize=(6 * (num_points+1), 6))

for ax, (x, y) in zip(axes[:num_points], points):
    for i in range(n_cols):
        imgs_rot180 = np.rot90(imgs[i], k=-1)
        ax.plot(b_s_mm2, imgs_rot180[x, y, idx_slc, :] / imgs_rot180[x, y, idx_slc, 0], 'o-', label=list(names_dict.keys())[i])
    ax.set_xlabel("b (s/mm$^2$)", **font_style)
    ax.set_ylabel("Signal Intensity", **font_style)
    ax.set_title(f"{list(points_dict.keys())[points.index((x, y))]} ({x}, {y}, 0)", **font_style)
    ax.legend(fontsize=14) 

ax = axes[-1]
ax.imshow(imgs_rot180[:, :, idx_slc, 0], cmap="gray")
for point in points:
    ax.plot(point[1], point[0], 'rx', markersize=12, markeredgewidth=2, label=f"{list(points_dict.keys())[points.index(point)]}")
    ax.annotate(f"{list(points_dict.keys())[points.index(point)]}", (point[1]+2, point[0]-2), color='red', fontsize=14, fontweight='bold')
ax.set_title("b=0 image with ROIs", **font_style)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"../figs/{''.join(names_dict.keys())}_decay.png", dpi=300)
plt.show()

# %%
names_dict = {
    "sense": "6_STEs_2mmiso_PA_sense_prelim_all_abs_pa.nii.gz",
    "4basis": "6_STEs_2mmiso_PA_4basis_bdelta0_bak_real_pa.nii.gz",
    "4basis_only1": "6_STEs_2mmiso_PA_4basis_bdelta0_subonly1_real_pa.nii.gz",
}
names = names_dict.values()
n_cols = len(names)

ss = [mdm.mdm_s_from_nii(p / name, 0) for name in names]
imgs = [nib.load(s.nii_fn).get_fdata() for s in ss]

# b_s_mm2 = [1000, 2000]
b_s_mm2 = ss[0].xps.b / 10**6
Nb = len(b_s_mm2)

font_style = {
    "fontsize": 16,
    "fontweight": "bold",
}
idx_slc = 0

fig, axes = plt.subplots(Nb, n_cols, figsize=(6*n_cols, 6*Nb))

for i in range(n_cols):
    for j in range(Nb):
        ax = axes[j, i]
        idx = np.where(np.isclose(ss[i].xps.b, b_s_mm2[j] * 10**6))[0][0]
        im = ax.imshow(np.rot90(imgs[i][:, :, idx_slc, idx], k=-1), cmap="gray")
        plt.colorbar(im)
        if j == 0:
            ax.set_title(f"{list(names_dict.keys())[i]}", pad=12, **font_style)
        if i == 0:
            ax.set_ylabel(f"b = {b_s_mm2[j]:.0f} s/mm$^2$", **font_style)

# axes[0, 0].set_ylabel("Magnitude ($I_0$)", **font_style)

for ax in axes.flatten():
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"../figs/{''.join(names_dict.keys())}.png", dpi=300)
plt.show()

# %% =====================signal decay w/o LTE======================
points_dict = {"csf": (46, 52), "wm": (40, 40), "gm": (30, 35)}
points = list(points_dict.values())
num_points = len(points)
fig, axes = plt.subplots(1, num_points+1, figsize=(6 * (num_points+1), 6))

for ax, (x, y) in zip(axes[:num_points], points):
    for i in range(n_cols):
        imgs_rot180 = np.rot90(imgs[i], k=-1)
        ax.plot(b_s_mm2, imgs_rot180[x, y, idx_slc, :] / imgs_rot180[x, y, idx_slc, 0], 'o-', label=list(names_dict.keys())[i])
    ax.set_xlabel("b (s/mm$^2$)", **font_style)
    ax.set_ylabel("Signal Intensity", **font_style)
    ax.set_title(f"{list(points_dict.keys())[points.index((x, y))]} ({x}, {y}, 0)", **font_style)
    ax.legend(fontsize=14) 

ax = axes[-1]
ax.imshow(imgs_rot180[:, :, idx_slc, 0], cmap="gray")
for point in points:
    ax.plot(point[1], point[0], 'rx', markersize=12, markeredgewidth=2, label=f"{list(points_dict.keys())[points.index(point)]}")
    ax.annotate(f"{list(points_dict.keys())[points.index(point)]}", (point[1]+2, point[0]-2), color='red', fontsize=14, fontweight='bold')
ax.set_title("b=0 image with ROIs", **font_style)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"../figs/{''.join(names_dict.keys())}_decay.png", dpi=300)
plt.show()

# %%

cmap='inferno'
worth_view_dict = {
  		"g_g": "lte_grappa_rot180_norm_removeb0stes_grappa_real_removeb0_bak",
		"s_s": "lte_sense_rot180_norm_removeb0stes_sens_abs_removeb0",
		"s_basis4": "lte_sense_rot180_norm_removeb0_stes_basis4_real_removeb0",
		# "s_basis4_ste": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0",
		"s_basis20_ste": "lte_sense_rot180_norm_removeb0_ste_basis20_real_removeb0",
		"s_basis4_stes_subonly1": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0subonly1",
		"s_basis4_ste_subonly1": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0_subonly1_bak",
}
worth_view = list(worth_view_dict.values())
directory_root = Path('/data/users/cyang/dtd_subspace/0119_ngc/processed/brain_2mmiso')
dtd_items_names_dict = {'dtd_gamma_MD':(0,4), 'dtd_gamma_Va':(0, 3), 'dtd_gamma_Vi':(0, 3)}
dtd_items_names = list(dtd_items_names_dict.keys()) 
n_rows = len(dtd_items_names)
names = worth_view
n_cols = len(names)

idx_slc = 0
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))

for i, name in enumerate(worth_view):
    src_dir = directory_root / name
    pattern = str(src_dir / '*.nii*')
    j = 0
    for fp in glob.glob(pattern):
        f = Path(fp)
        if any(k in f.name for k in dtd_items_names):
            k = [k for k in dtd_items_names if k in f.name]
            k = k[0]
            ax = axes[j, i]
            if j == 0:
                ax.set_title(f"{list(worth_view_dict.keys())[i]}", **font_style)
            if i == 0:
                ax.set_ylabel(f"{k}", **font_style)
            img = nib.load(f).get_fdata()
            img = np.rot90(img, k=1)
            vmin, vmax = dtd_items_names_dict[k]
            im = ax.imshow(img[:, :, 0], cmap=cmap, vmin=vmin, vmax=vmax)
            j += 1
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes.flatten():
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig(f"../figs/{''.join(worth_view_dict.keys())}_maps.png", dpi=300)
plt.show()
# %% !!paper
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np

# --- �������� ---
cmap = 'inferno'
worth_view_dict = {
    "STEs All Bases": "lte_sense_rot180_norm_removeb0stes_sens_abs_removeb0",
    "STEs 4 Bases": "lte_sense_rot180_norm_removeb0_stes_basis4_real_removeb0",
    "STE 20 Bases": "lte_sense_rot180_norm_removeb0_ste_basis20_real_removeb0",
    "STEs 4 Bases 1 Rep": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0subonly1",
    "STE 4 Bases 1 Rep": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0_subonly1_bak",
}
worth_view = list(worth_view_dict.values())
directory_root = Path('/data/users/cyang/dtd_subspace/0119_ngc/processed/brain_2mmiso')

# ��������MD, Vi, Va
dtd_items_names_dict = {
    'dtd_gamma_MD':(0,4),
    'dtd_gamma_Vi':(0, 3), 
    'dtd_gamma_Va':(0, 3)
}
dtd_items_names_dict_ylabel = {
    'dtd_gamma_MD': 'MD', 
    'dtd_gamma_Vi': "$V_I$", 
    'dtd_gamma_Va': '$V_A$'
}
dtd_items_names_ordered = list(dtd_items_names_dict.keys()) 

n_rows = len(dtd_items_names_ordered)
n_cols = len(worth_view)
idx_slc = 0

# --- ���� GridSpec ---
# ���������������������������������� colorbar ��
fig = plt.figure(figsize=(5 * n_cols + 1, 5 * n_rows))

# ����������������������������
# �� n_cols ������������ 1 (�������� MRI ��������)
# ������������ 0.05 (���������� colorbar ��������������)
# ���������� 0.1 (���������� colorbar)
width_ratios = [1] * n_cols + [0.05, 0.1]

# ������������������������ wspace �� hspace �� 0
gs = gridspec.GridSpec(n_rows, n_cols + 2, width_ratios=width_ratios, wspace=0.0, hspace=0.0)

font_style = {
    "fontsize": 24,
    "fontweight": "bold",
}

for j, metric_key in enumerate(dtd_items_names_ordered):
    im = None  # ������������������������
    
    for i, folder_name in enumerate(worth_view):
        src_dir = directory_root / folder_name
        pattern = str(src_dir / '*.nii*')
        matched_files = [f_path for f_path in glob.glob(pattern) if metric_key in Path(f_path).name]
        
        # ��������������������������
        ax = fig.add_subplot(gs[j, i])
        
        # ����������������
        if j == 0:
            ax.set_title(f"{list(worth_view_dict.keys())[i]}", **font_style)
        
        # ������������ Y ������
        if i == 0:
            ax.set_ylabel(f"{dtd_items_names_dict_ylabel[metric_key]}", **font_style)
            
        ax.set_xticks([])
        ax.set_yticks([])

        if matched_files:
            f = Path(matched_files[0])
            img = nib.load(f).get_fdata()
            img = np.rot90(img, k=1)
            vmin, vmax = dtd_items_names_dict[metric_key]
            
            im = ax.imshow(img[:, :, idx_slc], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            print(f"Warning: File for {metric_key} not found in folder {folder_name}")
            ax.axis('off')

    # ���������������� (gs[j, -1]) �������� colorbar
    if im is not None:
        cax = fig.add_subplot(gs[j, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=18) # ���� colorbar ������������

base_filenames = ''.join(worth_view_dict.keys()).replace(' ', '_')
save_path = f"../figs/paper_{base_filenames}_maps.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {save_path}")
plt.show()
# %%

cmap='coolwarm'
worth_view_dict = {
		"s_s": "lte_sense_rot180_norm_removeb0stes_sens_abs_removeb0",
		"s_basis4": "lte_sense_rot180_norm_removeb0_stes_basis4_real_removeb0",
		# "s_basis4_ste": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0",
		"s_basis20_ste": "lte_sense_rot180_norm_removeb0_ste_basis20_real_removeb0",
		"s_basis4_stes_subonly1": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0subonly1",
		"s_basis4_ste_subonly1": "lte_sense_rot180_norm_removeb0_ste_basis4_real_removeb0_subonly1_bak",
  		"g_g": "lte_grappa_rot180_norm_removeb0stes_grappa_real_removeb0_bak",
}
worth_view = list(worth_view_dict.values())
directory_root = Path('/data/users/cyang/dtd_subspace/0119_ngc/processed/brain')
dtd_items_names_dict = {'dtd_gamma_MD':(-4,4), 'dtd_gamma_Va':(-3, 3), 'dtd_gamma_Vi':(-3, 3)}
dtd_items_names = list(dtd_items_names_dict.keys()) 
n_rows = len(dtd_items_names)
names = worth_view
n_cols = len(names)

idx_slc = 0
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
ps_gold = directory_root / worth_view_dict["s_s"] 
gold_maps = {
    "dtd_gamma_MD" : nib.load(ps_gold / "dtd_gamma_MD.nii.gz").get_fdata(),
    "dtd_gamma_Va" : nib.load(ps_gold / "dtd_gamma_Va.nii.gz").get_fdata(),
    "dtd_gamma_Vi" : nib.load(ps_gold / "dtd_gamma_Vi.nii.gz").get_fdata(),
}
for i, name in enumerate(worth_view):
    src_dir = directory_root / name
    pattern = str(src_dir / '*.nii*')
    j = 0
    for fp in glob.glob(pattern):
        f = Path(fp)
        if any(k in f.name for k in dtd_items_names):
            k = [k for k in dtd_items_names if k in f.name]
            k = k[0]
            ax = axes[j, i]
            if j == 0:
                ax.set_title(f"{list(worth_view_dict.keys())[i]}", **font_style)
            if i == 0:
                ax.set_ylabel(f"{k}", **font_style)
            img = nib.load(f).get_fdata()
            img = np.rot90(img - gold_maps[k], k=1)
            vmin, vmax = dtd_items_names_dict[k]
            im = ax.imshow(img[:, :, 0], cmap=cmap, vmin=vmin, vmax=vmax)
            j += 1
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes.flatten():
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig(f"../figs/{''.join(worth_view_dict.keys())}_maps_residuals.png", dpi=300)
plt.show()
# %% ======================compare reconstructions======================
p = Path("/data/users/cyang/dtd_subspace/0119_ngc/DATA/brain_2mmiso/NII")
names_dict = {
    "STE_4basis_only1": "4_STE_2mmiso_PA_4basis_bdelta0_subonly1_coef.nii.gz",
    "STEs_4basis_only1": "6_STEs_2mmiso_PA_4basis_bdelta0_subonly1_coef.nii.gz",
    "STE_2basis_full": "4_STE_2mmiso_PA_2basis_bdelta0_bak_coef.nii.gz",
}
names = names_dict.values()
n_cols = len(names)

ss = [mdm.mdm_s_from_nii(p / name, 0) for name in names]
imgs = [nib.load(s.nii_fn).get_fdata() for s in ss]


font_style = {
    "fontsize": 16,
    "fontweight": "bold",
}
idx_slc = 0
num_basis = 4
# fig, axes = plt.subplots(Nb, n_cols, figsize=(6*n_cols, 6*Nb), gridspec_kw={"hspace": 0, "wspace": 0})
fig, axes = plt.subplots(num_basis, n_cols, figsize=(6*n_cols, 6*num_basis))

for i in range(n_cols):
    num_basis = imgs[i].shape[3]
    for j in range(num_basis):
        ax = axes[j, i]
        im = ax.imshow(np.rot90(imgs[i][:, :, idx_slc, j], k=-1), cmap="gray")
        plt.colorbar(im)
        if j == 0:
            ax.set_title(f"{list(names_dict.keys())[i]}", pad=12, **font_style)
        if i == 0:
            ax.set_ylabel(f"C$_{j+1}$", **font_style)

# axes[0, 0].set_ylabel("Magnitude ($I_0$)", **font_style)

for ax in axes.flatten():
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"../figs/{''.join(names_dict.keys())}_coeff.png", dpi=300)
plt.show()
# %% gemini
p = Path("/data/users/cyang/dtd_subspace/0119_ngc/DATA/brain_2mmiso/NII")

# Keep only the first item from the dictionary
full_names_dict = {
    "STE_4basis_only1": "4_STE_2mmiso_PA_4basis_bdelta0_subonly1_coef.nii.gz",
    "STEs_4basis_only1": "6_STEs_2mmiso_PA_4basis_bdelta0_subonly1_coef.nii.gz",
    "STE_2basis_full": "4_STE_2mmiso_PA_2basis_bdelta0_bak_coef.nii.gz",
}
first_key = list(full_names_dict.keys())[0]
names_dict = {first_key: full_names_dict[first_key]}

names = list(names_dict.values())
n_cols = len(names) # Will be 1

# Assuming mdm is already imported/available in your environment
ss = [mdm.mdm_s_from_nii(p / name, 0) for name in names]
imgs = [nib.load(s.nii_fn).get_fdata() for s in ss]

font_style = {
    "fontsize": 16,
    "fontweight": "bold",
}
idx_slc = 0
num_basis = imgs[0].shape[3]

# Create subplots for 1 column
# ... (previous data loading code)

# num_basis = imgs[0].shape[3]
fig, axes = plt.subplots(num_basis, 1, figsize=(6, 6 * num_basis), 
                         gridspec_kw={'hspace': 0, 'wspace': 0})

# Ensure axes is iterable even if num_basis is 1
if num_basis == 1:
    axes = [axes]

for j in range(num_basis):
    ax = axes[j]
    # Display image
    im = ax.imshow(np.rot90(imgs[0][:, :, idx_slc, j], k=-1), cmap="gray")
    
    # Remove all axis lines and ticks to make them seamless
    ax.axis("off") 
    
    # If you still want the "C_j" labels, use ax.text instead of ylabel 
    # because ylabel requires space that creates gaps.
    ax.text(0.02, 0.5, f"C$_{j+1}$", transform=ax.transAxes, 
            color="white", va="center", **font_style)

# Remove all margins
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)

plt.savefig(f"../figs/{list(names_dict.keys())[0]}_seamless.png", 
            dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
# %%
p = Path("/data/users/cyang/dtd_subspace/0119_ngc/DATA/brain/NII")
names_dict = {
    "STE_4basis_only1": "STEs_1_2mmiso_LR_full_5basis_bdelta0_16rep_noorder_coef.nii.gz",
}
names = names_dict.values()
n_cols = len(names)

ss = [mdm.mdm_s_from_nii(p / name, 0) for name in names]
imgs = [nib.load(s.nii_fn).get_fdata() for s in ss]


font_style = {
    "fontsize": 16,
    "fontweight": "bold",
}
idx_slc = 0
# fig, axes = plt.subplots(Nb, n_cols, figsize=(6*n_cols, 6*Nb), gridspec_kw={"hspace": 0, "wspace": 0})
fig, axes = plt.subplots(1,5, figsize=(6, 6*5))

num_basis = imgs[i].shape[3]
for j in range(num_basis):
    ax = axes[j]
    im = ax.imshow(np.rot90(imgs[0][:, :, idx_slc, j], k=-1), cmap="gray")
    # plt.colorbar(im)
    # if j == 0:
    #     ax.set_title(f"{list(names_dict.keys())[0]}", pad=12, **font_style)
    # if i == 0:
    #     ax.set_ylabel(f"C$_{j+1}$", **font_style)

# axes[0, 0].set_ylabel("Magnitude ($I_0$)", **font_style)

for ax in axes.flatten():
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])

# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"../figs/paper_{''.join(names_dict.keys())}_coeff.png", dpi=300)
plt.show()
# %%
