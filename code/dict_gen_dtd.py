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
from utils import ivim_model
from bart import bart

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


def get_dicc(f, D, Dstar, bvals, i):


    signal = ivim_model(f, D, Dstar, bvals)
    return i, signal, (f, D, Dstar)

def main():

    args = parser()
    # bvals = np.asarray([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000])
    from dipy.data import get_fnames
    from dipy.io.gradients import read_bvals_bvecs
    fraw, fbval,fbvec = get_fnames(name='ivim')
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

    size = 90
    fs = np.linspace(0, 0.4, size)  # 0, 0.4
    Ds = np.linspace(0.3e-3, 3e-3, size)  # try it
    Dstars = np.linspace(5e-3, 60e-3, size)
    size = int(size * size * size)
    ivim_dicc = np.zeros((size, len(bvals)))

    ## Code to generate the dictionary
    print("Now generating dictionary...")
    args2 = [(f, D, Dstar, bvals, i) for i, (f, D, Dstar) in enumerate(itertools.product(fs, Ds, Dstars))]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(get_dicc, args2)
        pool.close()
        pool.join()
        results = np.asarray(results, dtype=object)
        for i in range(results.shape[0]):
            ivim_dicc[i] = results[i, 1]

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

    return None


if __name__ == "__main__":

    main()

