# %%
import os
import numpy as np
import glob
import nibabel as nib
import matplotlib.pyplot as plt


def load_all_nii(directory):
	data = {}
	pattern = os.path.join(directory, '*.nii*')
	for path in sorted(glob.glob(pattern)):
		fname = os.path.basename(path)
		if fname.endswith('.nii.gz'):
			key = fname[:-7]
		elif fname.endswith('.nii'):
			key = fname[:-4]
		else:
			continue
		img = nib.load(path)
		data[key] = img.get_fdata()
	return data


if __name__ == "__main__":
	directory = '/data/users/cyang/dtd_subspace/0119_ngc/processed/brain/6_STEs_2mmiso_PA'
	# directory = '/data/users/cyang/dtd_subspace/0119_ngc/processed/brain/invivo_dtdgamma_subspace'
	data = load_all_nii(directory)
	for k, v in data.items():
		print(f"{k}: shape={v.shape}, dtype={v.dtype}")
	prefix = "dtd_gamma_"
	# prefix = "dtd_codivide_"
	# collect dtd_gamma_ images
	dtd_items = [(k, v) for k, v in data.items() if k.startswith(prefix)]
	n = len(dtd_items)
	print(f"{prefix} count: {n}")
	if n == 0:
		exit()
	cols = int(np.ceil(np.sqrt(n)))
	rows = int(np.ceil(n / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
	axes = axes.ravel()
	cmap='inferno'
	for ax, (k, v) in zip(axes, dtd_items):
		if v.ndim == 4:
			img2d = v[:, :, 0, 0]
		elif v.ndim == 3:
			img2d = v[:, :, 0]
		else:
			img2d = v
		im = ax.imshow(img2d, cmap=cmap)
		ax.set_title(k, fontsize=20)
		ax.axis("off")
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	for ax in axes[n:]:
		ax.axis("off")
	plt.tight_layout()
	plt.show()
# %%
data_phan = load_all_nii(directory)
dtd_phan_items = [(k, v) for k, v in data_phan.items() if k.startswith(prefix)]
n = len(dtd_phan_items)
print(f"{prefix} count: {n}")
fig, axes = plt.subplots(4, 2, figsize=(8, 16))
limits = [(0, 0.8), (0, 9000), (0, 0.9), (0, 0.9)]
for ax, ax2, (k, v), (k2, v2), (vmin, vmax) in zip(axes[:, 0], axes[:, 1], dtd_items, dtd_phan_items, limits):
	if v.ndim == 4:
		img2d = v[:, :, 0, 0]
	elif v.ndim == 3:
		img2d = v[:, :, 0]
	else:
		img2d = v
	im = ax.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax)
	if k == k2:
		im2 = ax2.imshow(v2, cmap=cmap, vmin=vmin, vmax=vmax) 
		fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
	ax.set_title(k, fontsize=16)
	ax.axis("off")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# %%
