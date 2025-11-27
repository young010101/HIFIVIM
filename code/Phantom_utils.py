import os
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from bart import bart
from sklearn.metrics import mean_squared_error
from utils import ivim_model



def calc_rmse(original_image, est_image, nrsme = True):

    rmse = np.sqrt(mean_squared_error(original_image.flatten(), est_image.flatten()))
    data_range = est_image.max()* 1.1 - est_image.min()*1.1
    ssim_index, _ = ssim(original_image, est_image, full=True, data_range=data_range)

    if nrsme:
        range_true = original_image.max() - original_image.min()
        nrmse = 100*(rmse/range_true)
        return np.round(nrmse,decimals=3), np.round(ssim_index, decimals=3)
    else:
        return np.round(rmse, decimals=4), np.round(ssim_index,decimals=4)


def get_other_metrics(original_image, est_image):

    differences = est_image - original_image
    med_bias_tmp = np.median(differences)
    med_error_tmp = np.median(np.abs(differences))
    median = np.median(est_image)
    mad = np.median(np.abs(est_image-median))
    rCV_tmp = mad / median


    return med_error_tmp, med_bias_tmp, rCV_tmp


def get_phantom(args):
    img_tmp = np.zeros((181, 217, 181))  # generate temporary data
    for file in os.listdir(args.phantom_dir):
        if '.DS' in file:
            pass
        elif 'gry' in file:
            img = nib.load(os.path.join(args.phantom_dir, file))
            data = img.get_fdata()
            img_tmp += data * 2
        elif 'wht' in file:
            img = nib.load(os.path.join(args.phantom_dir, file))
            data = img.get_fdata()
            img_tmp += data * 3
        elif 'csf' in file:  # csf
            img = nib.load(os.path.join(args.phantom_dir, file))
            data = img.get_fdata()
            img_tmp += data * 1
        else:
            pass
    plt.figure()
    plt.imshow(img_tmp[90, :,:])
    plt.show()

    return None


def make_phantom(args, show=True):
    slice = 24


    bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
    phantom = nib.load(os.path.join(args.outdir, 'Phantom_T1.nii.gz')).get_fdata()[..., slice]

    Dt, Fp, Dp = np.zeros_like(phantom), np.zeros_like(phantom), np.zeros_like(phantom)
    phantom_data = np.zeros((phantom.shape[0], phantom.shape[1], 15))
    WMmask, GMmask, CSFmask, WMH_mask1,WMH_mask2, BGmask = np.zeros_like(Dt), np.zeros_like(Dt), \
                                              np.zeros_like(Dt), np.zeros_like(Dt), np.zeros_like(Dt), np.zeros_like(Dt)

    center = (45,100)
    height, width = phantom.shape

    WMH_mask1 = np.zeros_like(Dt)
    y, x = np.ogrid[:height, :width]
    radius = 5
    WMH_mask1[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

    WMH_mask2 = np.zeros_like(Dt)
    center, radius = (53,60), 3.5
    WMH_mask2[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1

    WMH_mask3 = np.zeros_like(Dt)
    center, radius = (60,100), 3
    WMH_mask3[(x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2] = 1


    for i in range(phantom.shape[0]):
        for j in range(phantom.shape[1]):
            if WMH_mask1[i,j] == 1:
                Dt[i, j] = 0.0012
                Fp[i, j] = 0.16
                Dp[i, j] = 0.025
            elif WMH_mask2[i, j] == 1:
                Dt[i, j] = 0.0014
                Fp[i, j] = 0.17
                Dp[i, j] = 0.028
            elif WMH_mask3[i, j] == 1:
                Dt[i, j] = 0.0013
                Fp[i, j] = 0.165
                Dp[i, j] = 0.027
            else:
                if j < 101 and j > 65 and i > 62 and i < 103:
                    Dt[i, j] = 0.0006 if phantom[i, j] > 2.5 else 0.0005 if phantom[i, j] > 1.7 else 0.003 if phantom[
                                                                                                                  i, j] > 0 else 0
                    Fp[i, j] = 0.07 if phantom[i, j] > 2.5 else 0.06 if phantom[i, j] > 1.7 else 0.25 if phantom[
                                                                                                             i, j] > 0 else 0
                    Dp[i, j] = 0.045 if phantom[i, j] > 2.5 else 0.055 if phantom[i, j] > 1.7 else 0.02 if phantom[ i, j] > 0 else 0
                else:
                    Dt[i, j] = 0.0006 if phantom[i, j] > 2.35 else 0.0009 if phantom[i, j] > 1.7 else 0.003 if phantom[
                                                                                                                  i, j] > 0 else 0
                    Fp[i, j] = 0.07 if phantom[i, j] > 2.35 else 0.14 if phantom[i, j] > 1.7 else 0.2 if phantom[
                                                                                                            i, j] > 0 else 0
                    Dp[i, j] = 0.045 if phantom[i, j] > 2.35 else 0.03 if phantom[i, j] > 1.7 else 0.02 if phantom[
                                                                                                              i, j] > 0 else 0
            if Dt[i,j] == 0.0006:
                WMmask[i,j] = 1
            elif Dt[i, j] == 0.0005:
                BGmask[i, j] = 1
            elif Dt[i, j] == 0.0009:
                GMmask[i, j] = 1
            elif Dt[i, j] == 0.003:
                CSFmask[i, j] = 1


            phantom_data[i,j] = ivim_model(Fp[i,j], Dt[i,j], Dp[i,j], bvals)
            phantom_data[i,j][Dt[i,j] == 0] =0


    if show:
        plt.figure()
        plt.subplot(231), plt.imshow(np.rot90(WMmask), cmap='gray'), plt.axis('off')
        plt.subplot(232),  plt.imshow(np.rot90(GMmask), cmap='gray'), plt.axis('off')
        plt.subplot(233), plt.imshow(np.rot90(BGmask), cmap='gray'), plt.axis('off')
        plt.subplot(234), plt.imshow(np.rot90(WMH_mask1), cmap='gray'), plt.axis('off')
        plt.subplot(235), plt.imshow(np.rot90(WMH_mask2), cmap='gray'), plt.axis('off')
        plt.subplot(236), plt.imshow(np.rot90(WMH_mask3), cmap='gray'), plt.axis('off')

        cmap = 'turbo'
        fig, axes = plt.subplots(1, 4)
        axes[0].imshow(np.rot90(phantom), cmap='gray'), axes[0].set_xticks([]), axes[0].set_yticks([])
        axes[0].set_title('Phantom', fontsize=16, fontweight='bold')

        im = axes[1].imshow(np.rot90(Dt), cmap=cmap)
        axes[1].set_xticks([]), axes[1].set_yticks([])
        axes[1].set_title('D', fontsize=16, fontweight='bold'), im.set_clim(0.0003, 0.0015)
        cax = fig.add_axes([axes[1].get_position().x1 + 0.005,
                            axes[1].get_position().y0, 0.01, axes[1].get_position().height])
        cbar = plt.colorbar(axes[1].images[0], cax=cax)


        im = axes[2].imshow(np.rot90(Fp), cmap=cmap)
        axes[2].set_xticks([]), axes[2].set_yticks([])
        axes[2].set_title('f', fontsize=16, fontweight='bold'), im.set_clim(0.04, 0.2)
        cax = fig.add_axes([axes[2].get_position().x1 + 0.005,
                            axes[2].get_position().y0, 0.01, axes[2].get_position().height])
        cbar = plt.colorbar(axes[2].images[0], cax=cax)

        im = axes[3].imshow(np.rot90(Dp), cmap=cmap)
        axes[3].set_xticks([]), axes[3].set_yticks([])
        axes[3].set_title('D*', fontsize=16, fontweight='bold'), im.set_clim(0.01, 0.06)
        cax = fig.add_axes([axes[3].get_position().x1 + 0.005,
                            axes[3].get_position().y0, 0.01, axes[3].get_position().height])
        cbar = plt.colorbar(axes[3].images[0], cax=cax)

        plt.subplots_adjust(wspace=0.5)

        plt.figure()
        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.imshow(np.rot90(phantom_data[...,(i+3)*2]), cmap='gray'), plt.clim(0.2,1), plt.axis('off')
            plt.title("b = {} s/mm$^2$".format(bvals[(i+3)*2]), fontsize=14, fontweight='bold')

        plt.show()

    return Dt, Fp, Dp, phantom_data, (np.rot90(WMmask), np.rot90(GMmask), np.rot90(CSFmask),
                                      np.rot90(BGmask), np.rot90(WMH_mask1), np.rot90(WMH_mask2), np.rot90(WMH_mask3))


def add_phase(args, ivim, bvals = None, show=True, dki=False):
    shift_amount = int(-np.ceil(164 / (2 * 2)))
    bvals15 = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
    phase_estimates = np.roll(nib.load('Phantom/phase_estimates.nii.gz').get_fdata(dtype=np.complex128),
                                  shift_amount, axis=1)
    if bvals is None:
        bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]


    indices = [i for i, item in enumerate(bvals15) if item in (bvals)]
    phase_estimates = phase_estimates[...,indices]
    composite_ivim = np.zeros_like(ivim, dtype=np.complex128)

    for i in range(phase_estimates.shape[-1]):
        composite_ivim[..., i] = np.multiply(ivim[..., i],
                                             np.exp(1j * (np.angle(phase_estimates[..., i]))))

    if show:
        length = 13 if dki else 15
        fig1, axes1 = plt.subplots(4, length, figsize=(30, 10))  # composite phase
        fig2, axes2 = plt.subplots(2, length, figsize=(25, 10))
        for i in range(phase_estimates.shape[-1]):
            im = axes1[0, i].imshow(np.rot90(np.abs(composite_ivim[..., i]).squeeze()), cmap='gray')
            im.set_clim(0.2, 1)
            axes1[0, i].set_xticks([]), axes1[0, i].set_yticks([])
            axes1[0, i].set_title('b={} s/mm$^2$'.format(bvals[i]), fontweight='bold')

            im = axes1[1, i].imshow(np.rot90(np.angle(composite_ivim[..., i])), cmap='jet')
            im.set_clim(-np.pi, np.pi)
            axes1[1, i].set_xticks([]), axes1[1, i].set_yticks([])

            im = axes1[2, i].imshow(np.rot90(np.real(composite_ivim[..., i]).squeeze()), cmap='gray')
            # im.set_clim(0.2,1)
            axes1[2, i].set_xticks([]), axes1[2, i].set_yticks([])

            im = axes1[3, i].imshow(np.rot90(np.imag(composite_ivim[..., i]).squeeze()), cmap='jet')
            # im.set_clim(0.2,1)
            axes1[3, i].set_xticks([]), axes1[3, i].set_yticks([])
            if i == 0:
                axes1[0, i].set_ylabel('Magnitude', fontweight='bold')
                axes1[1, i].set_ylabel('Phase', fontweight='bold')
                axes1[2, i].set_ylabel('Real', fontweight='bold')
                axes1[3, i].set_ylabel('Imaginary', fontweight='bold')

        for i in range(phase_estimates.shape[-1]):
            im = axes2[0, i].imshow(np.rot90(np.abs(phase_estimates[..., i]).squeeze()), cmap='gray')
            # im.set_clim(0.2, 1)
            axes2[0, i].set_xticks([]), axes2[0, i].set_yticks([])
            axes2[0, i].set_title('b={} s/mm$^2$'.format(bvals[i]), fontweight='bold')

            im = axes2[1, i].imshow(np.rot90(np.angle(phase_estimates[..., i])), cmap='jet')
            im.set_clim(-np.pi, np.pi)
            axes2[1, i].set_xticks([]), axes2[1, i].set_yticks([])
            if i == 0:
                axes2[0, i].set_ylabel('Magnitude', fontweight='bold')
                axes2[1, i].set_ylabel('Phase', fontweight='bold')

        plt.show()

    return composite_ivim


def add_sens_maps(args, composite_ivim, bvals=None, show=True, dki=False):
    shift_amount = int(-np.ceil(164 / (2 * 2)))

    bvals15 = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
    if bvals is None:
        bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]

    indices = [i for i, item in enumerate(bvals15) if item in (bvals)]

    phase_estimates = np.roll(nib.load('Phantom/phase_estimates.nii.gz').get_fdata(dtype=np.complex128),
                                  shift_amount,
                                  axis=1)
    sens_maps = np.roll(nib.load('Phantom/sens_maps.nii.gz').get_fdata(dtype=np.complex128), shift_amount,
                            axis=1).squeeze()

    phase_estimates = phase_estimates[...,indices]

    composite_sens = np.zeros((sens_maps.shape[0], sens_maps.shape[1], sens_maps.shape[-1],
                               phase_estimates.shape[-1]), dtype=np.complex128)

    for i in range(composite_ivim.shape[-1]):
        for c in range(sens_maps.shape[-1]):
            tmp_mag = np.multiply(abs(composite_ivim[...,i]), abs(sens_maps[...,c]))
            tmp_phase = np.angle(sens_maps[...,c]) + np.angle(composite_ivim[...,i])
            tmp = tmp_mag * np.exp(1j*tmp_phase)
            composite_sens[:, :, c, i] = tmp

    if show:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig2, axes2 = plt.subplots(4,4,figsize=(10,10))
        for row in range(4):
            for col in range(4):
                axes[row, col].imshow(np.rot90(abs(sens_maps[...,row+col])),cmap='jet')
                axes[row,col].set_xticks([]),axes[row,col].set_yticks([])

                axes2[row, col].imshow(np.rot90(abs(composite_sens[:, :, row + col, 12])), cmap='gray')
                axes2[row, col].set_xticks([]), axes2[row, col].set_yticks([])


        plt.show()

    return composite_sens, sens_maps


def get_fft(args, composite_ivim_sens, show=True):

    fft_ivim = bart(1, 'fft 3', composite_ivim_sens)
    fft_ivim[:, ::2] = 0 # undersample R=2
    if show:
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig2, axes2 = plt.subplots(4, 4, figsize=(10, 10))
        for row in range(4):
            for col in range(4):
                im1 = axes[row, col].imshow(np.rot90(abs(fft_ivim[:,:, row + col, 9])**.2), cmap='gray')
                axes[row, col].set_xticks([]), axes[row, col].set_yticks([])

                im2 = axes2[row, col].imshow(np.rot90(abs(fft_ivim[:, :, row + col, 12])**.2), cmap='gray')
                axes2[row, col].set_xticks([]), axes2[row, col].set_yticks([])

                im1.set_clim(0, 5)
                im2.set_clim(0,5)

        plt.show()
    return fft_ivim



def pub_figure(basis2:list, basis3:list, basis4:list,  basis5:list, lowres:list, mags:list, GTs:list, dki=False):


    fig, axes = plt.subplots(3, 7, figsize=(10, 10))
    labels = ['D', 'f', 'D*']
    titles=['Ground-Truth', '2 Bases', '3 Bases', '4 Bases', '5 Bases', 'Phase Removal', 'Conventional']
    cmap='inferno'

    for row in range(axes.shape[0]):
        clim = (0.0003, 0.0015) if row == 0 else (0.04, 0.25) if row == 1 else (0.02, 0.06) if row == 2 else (0, 1.5)
        for col in range(axes.shape[1]):
            images = GTs if col == 0 else basis2 if col == 1 else basis3 if col == 2 else \
                basis4 if col == 3 else basis5 if col == 4 else lowres if col == 5 else mags
            im = axes[row, col].imshow(images[row],
                                       cmap=cmap)
            im.set_clim(clim)

            if col != 0:
                nrmse, ssim = calc_rmse(images[row], GTs[row], nrsme=True)

                text_to_add = "NRMSE: {}%\nSSIM: {}".format(nrmse, ssim)
                axes[row, col].text(82, 25, text_to_add, color='white', fontsize=12, fontweight='bold', ha='center')
            if row == 0:
                axes[row, col].set_title("{}".format(titles[col]), fontsize=20,
                                         fontweight='bold')
            if col == 0:
                axes[row,col].set_ylabel(labels[row], fontsize=24, fontweight='bold')

            axes[row,col].set_xticks([]), axes[row,col].set_yticks([])

            if col == 6:
                cbar_axes = fig.add_axes([axes[row, col].get_position().x1 + 0.02, axes[row, col].get_position().y0,
                                          0.01, axes[row,col].get_position().y1 - axes[row, col].get_position().y0])

                cbar = plt.colorbar(im, cax=cbar_axes)

    plt.subplots_adjust(wspace=0, hspace=0)

    return None


def bland_altman_image(basis2, basis3, basis4, basis5, mags, GTs, masks):

    fig, axes = plt.subplots(3,5, figsize=(10,10))
    labels=['D', 'f', 'D*']
    titles=['2 Bases', '3 Bases', '4 Bases', '5 Bases', 'Magnitude']
    ylims = [(-1e-4,2e-5), (0, 0.08), (-3e-3, 1.4e-1)]


    [axes[0, i].ticklabel_format(axis='both', style='sci', scilimits=(0, 0)) for i in range(5)]
    [axes[1, i].ticklabel_format(axis='both', style='sci', scilimits=(0, 0)) for i in range(5)]
    [axes[2, i].ticklabel_format(axis='both', style='sci', scilimits=(0, 0)) for i in range(5)]
    for row in range(3):
        loc = 'lower right' if row == 0 else 'upper right'
        print(labels[row].upper())
        for col in range(5):
            preds  =  basis2 if col == 0 else basis3 if col == 1 else basis4 if col == 2 else basis5 if col == 3 else mags
            print(titles[col].upper())
            calc_bland_altman_brain_phantom(map_ref=GTs[row], map_calc=preds[row], masks=masks,
                                            axis=axes[row,col],plot=True,name=labels[row], loc=loc)

            if row == 0:
                axes[row,col].set_title(titles[col], fontsize=16, fontweight='bold')

            if col!=0:
                axes[row,col].set_ylabel('')
                axes[row,col].set_yticks([])

    multiplier = 1.4
    ll, ul = min([axes[0, i].get_ylim()[0] for i in range(5)]), max([axes[0,i].get_ylim()[1] for i in range(5)])
    [axes[0, i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))  for i in range(5)]
    [axes[0, i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e')) for i in range(5)]
    [axes[0,i].set_ylim(ll*multiplier, ul*multiplier) for i in range(5)]

    ll, ul = min([axes[1, i].get_ylim()[0] for i in range(5)]), max([axes[1,i].get_ylim()[1] for i in range(5)])
    [axes[1, i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) for i in range(5)]
    [axes[1, i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e')) for i in range(5)]
    [axes[1,i].set_ylim(ll*multiplier, ul*multiplier) for i in range(5)]

    ll, ul = min([axes[2, i].get_ylim()[0] for i in range(5)]), max([axes[2,i].get_ylim()[1] for i in range(5)])
    [axes[2, i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) for i in range(5)]
    [axes[2, i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e')) for i in range(5)]
    [axes[2,i].set_ylim(ll*multiplier, ul*multiplier) for i in range(5)]

    plt.subplots_adjust()

    return None


def calc_bland_altman_brain_phantom(map_ref, map_calc, axis, masks, name, loc, fontsize=12, plot=False):
    """
    Calculates and plots a Bland-Altman plot for two methods, averaging over each ROI. The mean line
    provides an indication of the overall bias or systematic difference between the two methods.  The direction also
    indicates the direction of the bias - e.g. a negative bias indicates the second value underestimates compared to
    the GT.

    lines represent the 95% confidence interval. Could be due to systematic bias, an outlier, or a methodological issue.

    Parameters
    ----------
    map_ref: reference map, flattened, non-zeros
    map_calc:
    name:
    plot:

    Returns
    -------

    """
    # calculate difference
    map_calc2 =[]
    map_ref2 = []
    print('Median Error; Median Bias; rCV:\n')
    for mask in masks:

        med_error_tmp, med_bias_tmp, rCV_tmp = get_other_metrics(map_ref[mask==1], map_calc[mask==1])
        print('[{}, {}, {}]'.format(
            np.round(med_error_tmp,decimals=4), np.round(med_bias_tmp,decimals=4),
            np.round(rCV_tmp, decimals=4)
        ))
        # Only calculate mean if there are valid pixels in the mask
        mask_pixels = map_calc[mask==1]
        ref_pixels = map_ref[mask==1]
        if len(mask_pixels) > 0:
            map_calc2.append(mask_pixels.mean())
            map_ref2.append(ref_pixels.mean())
        else:
            # Skip this mask as it contains no valid data
            print(f"Warning: Mask contains no valid pixels, skipping...")
            continue

    map_calc2, map_ref2 = np.asarray(map_calc2), np.asarray(map_ref2)
    diff = (map_calc2 - map_ref2)
    mean = (map_calc2 + map_ref2)/ 2    # calculate average


    mean_diff, std_diff = np.nanmean(diff), np.nanstd(diff, ddof=1)
    lower_limit, upper_limit = mean_diff - 1.96*std_diff, mean_diff + 1.96*std_diff

    if plot:
        axis.scatter(mean, diff, marker='d', s = 75, label=f'Bias:\n{mean_diff:.4f}')
        axis.axhline(mean_diff, color='blue', linestyle='--')
        axis.axhline(lower_limit, color='red', linestyle='--')
        axis.axhline(upper_limit, color='red', linestyle='--')
        legend = axis.legend(loc=loc, markerscale=0, frameon=False)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        axis.set_xlabel("Average {}".format(name), fontsize=fontsize, fontweight='bold')
        axis.set_ylabel("Difference in {}".format(name), fontsize=fontsize, fontweight='bold')
        # Handle NaN values in mean array to avoid axis limit errors
        if len(mean[~np.isnan(mean)]) > 0:
            mean_finite = mean[~np.isnan(mean)]
            axis.set_xlim(mean_finite.min() - 0.125 * abs(mean_finite.min()),
                         mean_finite.max() + 0.125 * abs(mean_finite.max()))
        else:
            # If all values are NaN, set default limits
            axis.set_xlim(-1, 1)
    return None
