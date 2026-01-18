import numpy as np
import matplotlib.pyplot as plt

def plot_recon_vs_ivim(
    recon,
    ivim,
    bvals,
    points,
    recon_label="FMac",
    ivim_scale=1.0,
    figure_kwargs=None,
):
    recon_mag = np.real(recon.squeeze())
    if figure_kwargs is None:
        figure_kwargs = {}
    plt.figure(**figure_kwargs)
    colors = plt.cm.tab10(np.linspace(0, 1, len(points)))
    legend_entries = []
    for (i, j), c in zip(points, colors):
        plt.plot(bvals, recon_mag[i, j, :], marker="o", linestyle="-", color=c)
        plt.plot(bvals, ivim[i, j, :], marker="x", linestyle="--", color=c)
        legend_entries.append(f"{recon_label} ({i},{j})")
        legend_entries.append(f"dtd ({i},{j})")
    plt.xlabel("b-values (s/mm$^2$)")
    plt.ylabel("Signal Intensity")
    plt.title(f"{recon_label} Reconstructed Signal vs. Ground Truth IVIM Signal")
    plt.legend(legend_entries)