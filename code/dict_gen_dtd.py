#!/bin/python3
# Author: Alan Finkelstein
# email: alan_finkelstein@urmc.rochester.edu
# Department: Department of Biomedical Engineering, University of Rochester

import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib
import multiprocessing
import os
import cfl
from utils import dtd_gamma_model
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

description="Script to generate IVIM dictionary generation and basis estimation for reconstruction.\n" \
            "15 b-values are 0, 5, 7, 10, 15, 20, 30, 40,50, 60, 100, 200, 400, 700, 1000 "

font = { 
        # 'family' : 'normal', 
        'weight' : 'bold', 
        'size'   : 12
}

matplotlib.rc('font', **font)


def parser():

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--outdir", type=str, help='Directory to save dictionary to')
    parser.add_argument("--make_basis", type=bool, default=True)
    parser.add_argument("--basis_size", type=int, default=2)
    parser.add_argument("--show_figure", type=bool, default=True)

    args=parser.parse_args()

    return args


def get_dicc(s0, d_iso, mu2_iso, mu2_aniso, bvals, i):


    signal = dtd_gamma_model(s0, d_iso, mu2_iso, mu2_aniso, bvals)
    return i, signal, (s0, d_iso, mu2_iso, mu2_aniso)
def main():

    args = parser()
    bvals = np.asarray([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000, 1400, 2000])  # s/mm^2

    size = 90
    s0 = 10
    d_iso = np.linspace(0.001, 4, size)  # 0, 0.4
    mu2_iso = np.linspace(1e-6, 5, size)  # try it
    mu2_aniso = np.linspace(1e-6, 5, size)
    size = int(size * size * size * size)
    ivim_dicc = np.zeros((size, len(bvals)))

    ## Code to generate the dictionary
    print("Now generating dictionary...")
    args2 = [(s0, D, Dstar, mu2a, bvals, i) for i, (D, Dstar, mu2a) in enumerate(itertools.product(d_iso, mu2_iso, mu2_aniso))]
    start_time = datetime.datetime.now()
    print("Start time: ", start_time)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(get_dicc, args2)
        pool.close()
        pool.join()
        results = np.asarray(results, dtype=object)
        for i in range(results.shape[0]):
            ivim_dicc[i] = results[i, 1]
    end_time = datetime.datetime.now()
    print("End time: ", end_time)

    ### Code to Extract Basis Set #####
    if args.make_basis:
        print("Now extracting basis set ...")
        dicc = np.transpose(ivim_dicc, (1, 0))
        U, S, V = bart(3, 'svd -e', dicc)
        S1 = np.cumsum(S)
        S1 = S1 / S1.max()
        basis = bart(1, 'extract 1 0 {}'.format(args.basis_size), U)  # extract basis
        basis = bart(1, 'transpose 1 6', basis)  # place into correct bart format
        basis = bart(1, 'transpose 0 5', basis)

        cfl.writecfl(os.path.join(args.outdir, 'ivim_basis_{}'.format(args.basis_size)), (basis))

        # Visualize Dictionary and Basis
        if args.show_figure:
            plt.figure()
            n = 5000
            plt.plot(bvals, (abs(ivim_dicc[::n].T)))
            plt.xlabel('b-values [s/mm$^2$]', fontsize=22, fontweight='bold')
            plt.ylabel('Signal Intensity [AU]', fontsize=22, fontweight='bold')

            plt.figure()
            plt.subplot(121), plt.plot(abs(S1[:])), plt.scatter(np.arange(0, len(S1)), abs(S1[:]))
            plt.xlabel('Principal Component', fontsize=22, fontweight='bold')
            plt.ylabel('Percentage', fontsize=22, fontweight='bold')

            plt.subplot(122), plt.plot(bvals, (np.squeeze(np.real(basis))))
            plt.xlabel('b-values [s/mm$^2$]', fontsize=22, fontweight='bold')
            plt.ylabel('Signal Intensity [AU]', fontsize=22, fontweight='bold')
            plt.legend(['$\Phi_1$', '$\Phi_2$', '$\Phi_3$'], fontsize=18)
            plt.show()

    print("Basis set generation complete. Use ivim_basis_{} for LLR + Subspace reconstruction".format(
        args.basis_size))
    print(f"ended in {datetime.datetime.now() - start_time}")

    return None


if __name__ == "__main__":

    main()

