import datetime
import numpy as np
import multiprocessing
import corner
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from skimage.metrics import structural_similarity as ssim
import os
try:
    from bart import bart
except ModuleNotFoundError:
    import sys
    # Try to locate BART's Python bindings via environment hints
    bart_python_env = os.getenv("PYTHONPATH")
    bart_toolbox = os.getenv("BART_TOOLBOX_PATH")

    candidates = []
    if bart_python_env:
        candidates.append(bart_python_env)
    if bart_toolbox:
        candidates.append(os.path.join(bart_toolbox, "python"))
        candidates.append(os.path.join(bart_toolbox, "python3"))

    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from bart import bart
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot import 'bart'. Ensure BART Python path is on PYTHONPATH or BART_TOOLBOX_PATH is set. Tried: "
            + ", ".join([str(x) for x in candidates])
        ) from e
from bart import bart
from scipy.optimize import nnls
from dipy.segment.mask import median_otsu
from scipy.ndimage import map_coordinates
from scipy import stats
from sklearn.metrics import mean_squared_error
from skimage.registration import phase_cross_correlation

'''
Author: Alan Finkelstein. 
Institution: University of Rochester 
Department: Department of Biomedical Engineering 
Date: 5/4/2023 

Functions for model-based IVIM/IVIM-LLR
'''


# TODO: add documentation and commenting for everything.


def ivim_model(f: float, D: float, Dstar: float, bvals):
    """
    :param f: perfusion fraction
    :param D: ADC mm/s2
    :param Dstar: pseudodiffusion coefficient mm/s2
    :return:
    """
    tmp = []
    for i, b in enumerate(bvals):
        S = f * np.exp(-b * Dstar) + (1 - f) * np.exp(-b * D)  #
        tmp.append(S)
    tmp = tmp / tmp[0]
    return tmp


def dtd_gamma_model(
    s0,
    d_iso,
    mu2_iso,
    mu2_aniso,
    bvals,
    b_delta=None,
    b_eta=None,
    rs=None,
    s_ind=None,
):
    """
    Python equivalent of dtd_gamma_1d_fit2data

    Parameters
    ----------
    bvals : array
        b-values (s/mm^2)
    s0 : float
        Baseline signal
    d_iso : float
        Mean diffusivity (MD)
    mu2_iso : float
        Isotropic variance
    mu2_aniso : float
        Anisotropic variance
    b_delta : array or None
        b-tensor anisotropy
    b_eta : array or None
        Asymmetry parameter
    rs : array or None
        Relative signal scaling across series
    s_ind : array or None
        Series index

    Returns
    -------
    s : array
        Signal S(b)
    """

    bvals = np.asarray(bvals) * 1e-3  # convert to s/um^2

    # ---- baseline weighting (series-dependent S0) ----
    if rs is not None and s_ind is not None:
        rs = np.asarray([1.0] + list(rs))
        sw = s0 * np.sum(
            (rs[None, :] * (s_ind[:, None] == np.arange(1, len(rs) + 1))),
            axis=1,
        )
    else:
        sw = s0

    # ---- total diffusional variance ----
    if b_delta is None:
        mu2 = mu2_iso
    else:
        if b_eta is None:
            b_eta = 0
        mu2 = mu2_iso + mu2_aniso * b_delta**2 * (b_eta**2 + 3) / 3

    # ---- gamma model signal ----
    s = sw * (1 + bvals * mu2 / d_iso) ** (-d_iso**2 / mu2)

    return np.real(s)


def phase_estimate_(complexdata, r=25):
    ham = np.hamming(complexdata.shape[0])[:, None]
    ham2 = np.hamming(complexdata.shape[1])[None,:]

    ham2D = np.sqrt(np.dot(ham, ham2)) ** r
    ksp = np.fft.fftshift(np.fft.fft2(complexdata))
    ksp_filtered = ham2D * ksp
    return np.fft.ifft2(np.fft.fftshift(ksp_filtered))


def lowres_phaseremoval(complexdata, r=25):
    """
    :param complexdata: of size - rows, columns, bvalues - should be a single slice.
    :return:
        corrected data with low res phase removal
    """
    complex2 = np.zeros_like(complexdata, dtype=np.complex128)
    for b in range(complexdata.shape[-1]):
        phase_estimate = phase_estimate_(complexdata[..., b])
        corrected = np.multiply(complexdata[..., b], np.exp(-1j * np.angle(phase_estimate)))
        complex2[..., b] = corrected

    return complex2


def ivim_fit_segmented(data, mask, bvals=None):

    if bvals is None:
        bvals = np.asarray([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000])
    else:
        bvals = bvals

    data = data/data[0]
    if np.isnan(data).any():
        mask2 = 0
    else:
        mask2 = 1
    if mask == 1 and mask2 == 1:
        if len(bvals)>12:
            high_b = bvals[bvals >= 400]
            high_signal = data[bvals >= 400]
            params, _ = curve_fit(lambda bval, Dt: np.exp(-bval * Dt), high_b, high_signal, p0=(0.001),
                              bounds=(0,0.005), maxfev=5000)

            Dt2 = params[0]
            bounds = ([0.001, 0.005, Dt2/10], [.1, 0.5, 0.01])
            params, _ = curve_fit(lambda bval, Dp,Fp, Dt: Fp * np.exp(-bval * Dp) + (1-Fp)*np.exp(-bval*Dt), bvals,
                              data, p0=(0.01, 0.1, Dt2), bounds=bounds, maxfev=10000)

            Dp, Fp, Dt= params[0], params[1], params[2]
        else:

            bounds = ([0, 0.01, 0], [0.04, 0.3, 0.004])
            params, _ = curve_fit(lambda bval, Dp,Fp,Dt: Fp * np.exp(-bval * Dp) + (1-Fp)*np.exp(-bval*Dt), bvals,
                              data, p0=(0.01, 0.1,0.001), bounds=bounds, maxfev=10000)

            Dp, Fp, Dt = params[0], params[1], params[2]

        return order(Dt, Dp, Fp)
    else:
        return 0, 0, 0

def ivim_fit_bayes(data, mask, nwalkers=75, lenb=15, plot=False):
    # if fit segmented is working again, then can use that as prior.
    # np.round(Dt, 8), np.round(Dp, 8), np.round(Fp, 8)

    bval = np.asarray([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000])
    data = data / data[0]
    if mask == 1:
        tmp1 = np.random.uniform(0.3e-3, 3e-3, (nwalkers, 1))
        tmp2 = np.random.uniform(0, 0.25, (nwalkers, 1))
        tmp3 = np.random.uniform(2e-3,60e-3, (nwalkers, 1))
        p0 = np.concatenate([tmp1, tmp2, tmp3], axis=1)
        try:
            sampler = emcee.EnsembleSampler(nwalkers, 3, log_probability, args=(bval, data))
            sampler.run_mcmc(p0, 100, progress=False)
            samples = sampler.get_chain(discard=10, thin=2, flat=True)
            Dt, Fp, Dp = np.median(samples[:, 0]), np.median(samples[:, 1]), np.median(samples[:, 2])
            if plot:
                try:
                    plt.figure()
                    fig = corner.corner(samples, labels=['Dt', 'Fp', 'Dp'],
                                range=[(1e-4, 3e-3), (0, 0.4), (0, 50e-3)], truths=[Dt, Fp, Dp])
                    plt.show()
                except:
                    pass

            return order(Dt, Dp, Fp)

        except ValueError:
            return 0., 0., 0.
    else:
        return 0,0,0


# Define the log-prior function
def log_prior(theta):
    D, f, D_star = theta
    if 0.0001 < D < 0.1 and 0.001 < f < 0.999 and 0.001 < D_star < 0.1:
        return 0.0
    return -np.inf


# Define the log-posterior function
def log_probability(theta, b, S):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(theta, b, S)


# Define the log-likelihood function
def log_likelihood(theta, b, S):
    # print(theta.shape)
    D, f, D_star = theta
    S_model = (f * np.exp(-b * (D_star)) + (1 - f) * np.exp(-b * (D)))
    sigma = np.sqrt(np.sum((S - S_model) ** 2) / (len(S) - len(theta)))

    return -0.5 * np.sum((S - S_model) ** 2 / sigma ** 2 + np.log(sigma))


def order(Dt, Dp, Fp):
    if Dp < Dt:
        Dp, Dt = Dt, Dp
    if Fp > 0.5:
        Fp = 1 - Fp
    return np.round(Dt, 8), np.round(Dp, 8), np.round(Fp, 8)


def get_initial_sens(data,sens):

    """
    :param data: raw k-space data
    :param sens: sensitivity maps
    """
    PhaseShiftBase = np.pi
    peshift = np.ceil(2*np.pi/PhaseShiftBase)
    shift_amount = int(-np.ceil(164/(2*2)))

    sens_prelim = np.zeros((data.shape[0], data.shape[1], data.shape[-1]), dtype=np.complex128)

    for i in range(data.shape[-1]): # 300
        sens_prelim[..., i] = bart(1, 'pics -e -d 5 -i 100 -S -R L:3:3:0.001 -R W:3:0:0.001', data[...,i], sens)

    return sens_prelim


def get_composite_sens(data, sens_maps, bvals=None, visualize="False", title=''):
    """
    :param complex_data: - of size (NxNx1x67)
    :param sens_maps: preliminary sens maps (NxNx1x16)

    For 2D data right now
    """
    # e.g. 164 x 164 x 16 x 67
    bvals1 = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
    if bvals is None:
        bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
    indices = [i for i, item in enumerate(bvals1) if item in (bvals)]

    composite_sens = np.zeros((data.shape[0], data.shape[1], sens_maps.shape[-1], data.shape[-1]),
                              dtype=np.complex128)
    shift_amount = int(-np.ceil(164 / (2 * 2)))
    phase_estimates = np.zeros((data.shape[0], data.shape[1], 15), dtype=np.complex128)
    phase_estimates = phase_estimates[...,indices]
    for i in range(data.shape[-1]):
        phase_estimate = phase_estimate_(data.squeeze()[:, :, i])  # estimate low resolution phase
        phase_estimates[:,:,i] = phase_estimate
        for c in range(sens_maps.shape[-1]):
            composite_sens[:, :, c, i] =np.multiply(sens_maps[:, :, 0,c], #328x164 * 328x164
                                                         np.exp(1j * np.angle(phase_estimate)))
            if visualize=="True":
                if (i % 2 == 0) and c == 3:

                    plt.figure()
                    plt.subplot(231), plt.imshow(np.abs(np.rot90(np.roll(sens_maps[:, :,0, c],shift_amount, axis=1))), cmap='jet'), plt.axis('off')
                    plt.subplot(232), plt.imshow(np.angle(np.rot90(np.roll(sens_maps[:, :, 0, c],shift_amount,axis=1))), cmap='jet'), plt.axis('off')
                    plt.subplot(233), plt.imshow(np.angle(np.rot90(np.roll(phase_estimate[:],shift_amount,axis=1))), cmap='jet'), plt.axis('off')
                    plt.subplot(234), plt.imshow(np.abs(np.rot90(np.roll(composite_sens[:, :,  c, i], shift_amount, axis=1))), cmap='jet'), plt.axis('off')
                    plt.subplot(235), plt.imshow(np.angle(np.rot90(np.roll(composite_sens[:, :, c, i], shift_amount, axis=1))), cmap='jet'), plt.axis('off')
                    plt.suptitle("bvalue: {}, Channel: {}, Direction: {}".format(bvals[i],c, title))
                    plt.show()

    composite_sens = np.expand_dims(composite_sens, axis=4)

    return composite_sens, phase_estimates


def llr_recon(data, sens, basis, R=2, lambda1=0.005, lambda2=0.001, use_basis=False):
    """
    :param data: should be a 2D data slice.
    :param sens:
    :param lambda_ : regularization weight
    :param basis: basis set derived from ivim dictionary for '45 direction'
    :param use_basis: boolean - whether or not to use basis function.
    """

    if use_basis:
        prod = data.shape[3]*data.shape[-1]
        tmp1 = bart(1, 'fmac', sens, basis)
        tmp2 = bart(1, 'reshape 40 {} 1'.format(prod),  tmp1)
        multisens_stacked = bart(1, 'transpose 4 6', tmp2)
        data_stacked = bart(1, 'reshape 40 {} 1'.format(prod), data)
        print(data.shape, basis.shape, sens.shape)

        if R==3:
            recon = bart(1, 'pics -d 5 -i 100 -S -c -R L:3:3:0.00005 -R W:3:0:0.00001', data_stacked, multisens_stacked)
        else:
            print(f"data_stacked.shape: {data_stacked.shape}, multisens_stacked.shape: {multisens_stacked.shape}")
            recon = bart(1, 'pics -i 100 -d 5 -e -S -R L:3:3:{} -R W:3:0:{}'.format(lambda1, lambda2), data_stacked, multisens_stacked)

        recon = bart(1, 'transpose 4 6', recon)
        recon_fmac = bart(1, 'fmac -s 64', basis, recon)

        return recon, recon_fmac
    else:
        if R==3:
            recon = bart(1, 'pics -d 5 -i 100 -S -R L:3:3:0.00005 -R W:3:0:0.00001', data, sens)
        else:
            recon = bart(1, 'pics -d 5 -i 300 -S -R L:3:3:0.0001 -R W:3:0:0:0005', data, sens)

        return recon


def slice_recon(data, bvals, mask, kind='ivim'):

    print(multiprocessing.cpu_count())
    num_processes = multiprocessing.cpu_count()

    if kind == 'ivim':
        with multiprocessing.Pool(processes=num_processes) as pool:
            args = [(data[i, j, k], mask[i, j, k], bvals) for i in range(data.shape[0])
                    for j in range(data.shape[1]) for k in range(data.shape[2])]
            start = datetime.datetime.now()
            results = pool.starmap(ivim_fit_segmented, args)  # results = pool.starmap(ivim_fit_bayes, args)
            results = np.asarray(results).reshape((164, 164, 70, 3))
            Dt, Dp, Fp = results[..., 0], results[..., 1], results[..., 2]
            print(datetime.datetime.now() - start)

            return Dt * mask, Fp * mask, Dp * mask, np.zeros_like(Dt)
    else:
        return None


def add_noise(data, SNR, bvals = None, return_img=True, return_clean=False, dki=False):
    """
    Adds complex Gaussian white noise to k-space data with a specific SNR.

    Parameters
    ----------
    data (complex ndarray): Input complex k-space data
    SNR (float or int): Desired SNR in dB (e.g. 20 dB).
    return_img (boolean): default is True, determines whether to return data in image space or k-space.

    Returns
    -------
    returns nd-array corrupted by gaussian white noise in k-space; returns either data in image space or kspace.
    """

    image_shape = data[0,0].shape

    if bvals is None:
        bvals  = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
    SNRs = np.interp(bvals, [0, 1000], [SNR, 10])

    for k, b in enumerate(bvals):
        P = np.mean(np.abs(data[..., k]) ** 2, axis=(0, 1))
        N = P / (10 ** (SNRs[k] / 10))  # noise power
        noise = np.random.normal(0, np.power(N, 1 / 2), (data.shape[0], data.shape[1])) + \
                1j * np.random.normal(0, np.power(N, 1 / 2), (data.shape[0], data.shape[1]))
        data[..., k] += noise

    if return_img:
        return bart(1, 'fft -i 3', data)
    elif return_clean:
        return data
    else:
        return data

def calc_bland_altman(map_ref, map_calc, name, xlim, fontsize=12, plot=False):
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
    phantom2 = bart(1, 'phantom -x 128 -T -b -s {}'.format(1))
    for i in range(11):
        map_calc2.append(map_calc[abs(phantom2[...,i].squeeze() ==1)].mean())
        map_ref2.append(map_ref[abs(phantom2[...,i].squeeze() == 1)].mean())
    map_calc2, map_ref2 = np.asarray(map_calc2), np.asarray(map_ref2)
    diff = (map_calc2 - map_ref2)
    mean = (map_calc2 + map_ref2)/ 2

    diff[map_calc2 ==0] = np.nan
    mean[map_calc2 ==0] = np.nan
    mean_diff = np.nanmean(diff)
    std_diff = np.nanstd(diff, ddof=1)

    lower_limit = mean_diff - 1.96*std_diff
    upper_limit = mean_diff + 1.96*std_diff


    if plot:

        plt.scatter(mean, diff, marker='d', s = 75)
        plt.axhline(mean_diff, color='blue', linestyle='--', label=f'MEAN:\n{mean_diff:.5f}')
        plt.axhline(lower_limit, color='red', linestyle='--')
        plt.axhline(upper_limit, color='red', linestyle='--')
        plt.xlabel("Average {}".format(name), fontsize=fontsize, fontweight='bold'), \
        plt.ylim(xlim)
        plt.ylabel("Difference in {}".format(name), fontsize=fontsize, fontweight='bold')
        plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        plt.legend(loc='upper right')
    return None

def normalize(data):

    return (data - data.min())/(data.max() - data.min())

def calc_rmse(original_image, est_image, nrsme=True):
    rmse = np.sqrt(mean_squared_error(original_image.flatten(), est_image.flatten()))
    data_range = est_image.max() * 1.1 - est_image.min() * 1.1
    ssim_index, _ = ssim(original_image, est_image, full=True, data_range=data_range)

    if nrsme:
        range_true = original_image.max() - original_image.min()
        nrmse = 100 * (rmse / range_true)
        return np.round(nrmse, decimals=3), np.round(ssim_index, decimals=3)
    else:
        return np.round(rmse, decimals=4), np.round(ssim_index, decimals=4)

