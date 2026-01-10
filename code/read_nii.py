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
	directory = '/data/users/cyang/repos/HIFIVIM/Phantom/phan_cyan/processed/brain'
	data = load_all_nii(directory)
	for k, v in data.items():
		print(f"{k}: shape={v.shape}, dtype={v.dtype}")

	# collect dtd_gamma_ images
	dtd_items = [(k, v) for k, v in data.items() if k.startswith("dtd_gamma_")]
	n = len(dtd_items)
	print(f"dtd_gamma_ count: {n}")
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
		ax.set_title(k)
		ax.axis("off")
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	for ax in axes[n:]:
		ax.axis("off")
	plt.tight_layout()
	plt.show()
# %%
