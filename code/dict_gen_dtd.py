#!/bin/python3
# Author: Cheng Yang 
# email: yc5@tju.edu.cn
# Department: Academy of Medical Engineering and Translational Medicine, Tianjin University

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib
import multiprocessing
import os
import cfl
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
from utils import dtd_gamma_model, BVALS
from functools import wraps

description="Script to generate IVIM dictionary generation and basis estimation for reconstruction.\n" \
            "15 b-values are 0, 5, 7, 10, 15, 20, 30, 40,50, 60, 100, 200, 400, 700, 1000 "

font = { 
        # 'family' : 'normal', 
        'weight' : 'bold', 
        'size'   : 12
}

matplotlib.rc('font', **font)


def parser(args=None):

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", type=str, help='Directory to save dictionary to')
    parser.add_argument("--make_basis", type=bool, default=True)
    parser.add_argument("--basis_size", type=int, default=2)
    parser.add_argument("--show_figure", type=bool, default=True)

    args=parser.parse_args(args)

    return args


def get_dicc(s0, d_iso, mu2_iso, mu2_aniso, bvals, i):


    signal = dtd_gamma_model(s0, d_iso, mu2_iso, mu2_aniso, bvals)
    return i, signal, (s0, d_iso, mu2_iso, mu2_aniso)

def get_dicc2(s0, d_iso, mu2_iso, mu2_aniso, bvals, i):
    signal = dtd_gamma_model(s0, d_iso, mu2_iso, mu2_aniso, bvals, b_delta=np.ones(len(bvals)))
    return i, signal, (s0, d_iso, mu2_iso, mu2_aniso)


args = parser(["--outdir", "../Standard_Files_dtd", "--basis_size", "3"])
# bvals = np.asarray([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000, 1400, 2000])  # s/mm^2
# bvals = np.asarray([0, 7, 10, 15, 20, 40, 50, 60, 100, 200, 400, 700, 1000, 1400, 2000])  # s/mm^2

size = 90
s0 = 1
d_iso = np.linspace(0.1, 3.5, size)  # 0, 0.4
mu2_iso = np.linspace(1e-6, 2, size)  # try it
mu2_aniso = np.linspace(1e-6, 2, size)



def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__name__}] time cost: {end - start:.6f} s")
        return result
    return wrapper
    

@timeit
def generate_dictionary(bvals = BVALS):
    """generate dictionary

    Args:
        bvals (list, optional): b values. Defaults to BVALS.

    Returns:
        numpy.array: num_dict * num_bvals
    """
    dict_size = int(size * size * size)
    ivim_dicc = np.zeros((dict_size, len(bvals)))
    ## Code to generate the dictionary
    print("=========Now generating dictionary...==========")
    args2 = [(s0, D, Dstar, mu2a, bvals, i) for i, (D, Dstar, mu2a) in enumerate(itertools.product(d_iso, mu2_iso, mu2_aniso))]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Use b_delta = 1 (tensor anisotropy) version of the model
        results = pool.starmap(get_dicc2, args2)
        pool.close()
        pool.join()
        results = np.asarray(results, dtype=object)
        for i in range(results.shape[0]):
            ivim_dicc[i] = results[i, 1]

    # %% ensure dictionary validity
    print(f"Dictionary shape: {ivim_dicc.shape}")
    return ivim_dicc 


@timeit
def extract_basis_set(ivim_dicc, num_basis, debug=False, bvals=BVALS):
    print("=======Now extracting basis set ...")
    dicc = np.transpose(ivim_dicc, (1, 0))
    U, S, V = bart(3, 'svd -e', dicc)
    S1 = np.cumsum(S)
    S1 = S1 / S1.max()
    if debug:
        print(f"Dictionary shape for SVD: {dicc.shape}") 
        print(f"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
        print(f"Type of dictionary: {type(dicc)}, Type of U: {type(U)}") 
    basis = bart(1, 'extract 1 0 {}'.format(num_basis), U)  # extract basis
    basis = bart(1, 'transpose 1 6', basis)  # place into correct bart format
    basis = bart(1, 'transpose 0 5', basis)
    if debug:
        plot_S1_basis(S1, bvals, basis)     
    return basis, S1


@timeit
def basis_pipeline(num_basis, bvals, outdir=None, debug=False):
    ivim_dicc = generate_dictionary(bvals=bvals)
    basis, _S1 = extract_basis_set(ivim_dicc, num_basis,bvals=bvals, debug=debug) 
    if outdir is not None:
        cfl.writecfl(os.path.join(outdir, 'dtd_bdelta1_basis_{}'.format(num_basis)), (basis))
    return basis


def _plot_S1(ax, S1):
    ax.plot(abs(S1[:]))
    ax.scatter(np.arange(0, len(S1)), abs(S1[:]))
    ax.set_xlabel('Principal Component', fontsize=22, fontweight='bold')
    ax.set_ylabel('Percentage', fontsize=22, fontweight='bold')


def _plot_basis(ax, bvals, basis): 
    ax.plot(bvals, (np.squeeze(np.real(basis))))
    ax.set_xlabel('b-values [s/mm$^2$]', fontsize=22, fontweight='bold')
    ax.set_ylabel('Signal Intensity [AU]', fontsize=22, fontweight='bold')
    ax.legend(['$\Phi_1$', '$\Phi_2$', '$\Phi_3$'], fontsize=18)


def plot_S1_basis(S1, bvals, basis):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    _plot_S1(ax, S1)

    ax = axes[1]
    _plot_basis(ax, bvals, basis) 
    fig.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    ivim_dicc = generate_dictionary()
    # %%
    ### Code to Extract Basis Set #####
    if args.make_basis:
        basis, S1 = extract_basis_set(ivim_dicc, args.basis_size) 
        # todo
        bvals = BVALS
        cfl.writecfl(os.path.join(args.outdir, 'dtd_bdelta1_basis_{}'.format(args.basis_size)), (basis))
        # Visualize Dictionary and Basis
        if args.show_figure:
            plt.figure()
            n = 5000
            plt.plot(bvals, (abs(ivim_dicc[::n].T)))
            plt.xlabel('b-values [s/mm$^2$]', fontsize=22, fontweight='bold')
            plt.ylabel('Signal Intensity [AU]', fontsize=22, fontweight='bold')

            plot_S1_basis(S1, bvals, basis) 
    # %%
    print("Basis set generation complete. Use dtd_bdelta1_basis_{} for LLR + Subspace reconstruction".format(
        args.basis_size))
