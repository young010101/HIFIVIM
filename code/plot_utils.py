import numpy as np
import matplotlib.pyplot as plt

def plot_recon_vs_ivim(
    img_arr,
    bvals,
    points,
    recon_label="FMac",
    ivim_scale=1.0,
    figure_kwargs=None,
):
    if isinstance(img_arr, dict):
        img_items = list(img_arr.items())
    elif isinstance(img_arr, (list, tuple)):
        if len(img_arr) > 0 and isinstance(img_arr[0], (list, tuple)) and len(img_arr[0]) == 2:
            img_items = list(img_arr)
        else:
            img_items = [(f"{recon_label}-{idx + 1}", arr) for idx, arr in enumerate(img_arr)]
    else:
        img_items = [(recon_label, img_arr)]

    if figure_kwargs is None:
        figure_kwargs = {}
    plt.figure(**figure_kwargs)
    colors = plt.cm.tab10(np.linspace(0, 1, len(points)))
    legend_entries = []
    for (i, j), c in zip(points, colors):
        for idx, (label, img_3d) in enumerate(img_items):
            recon_mag = np.real(img_3d.squeeze())
            marker = ["o", "s", "^", "D", "v", "P", "X"][idx % 7]
            linestyle = ["-", "--", "-.", ":"][idx % 4]
            plt.plot(bvals, recon_mag[i, j, :], marker=marker, linestyle=linestyle, color=c)
            legend_entries.append(f"{label} ({i},{j})")
    plt.xlabel("b-values (s/mm$^2$)")
    plt.ylabel("Signal Intensity")
    plt.title(f"{recon_label} Reconstructed Signal vs. Ground Truth IVIM Signal")
    plt.legend(legend_entries)
