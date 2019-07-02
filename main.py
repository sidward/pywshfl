#!/usr/bin/env python3

# -------------------------------------------------------------------------------------------------------------------------- #

import time
import numpy as np

# -------------------------------------------------------------------------------------------------------------------------- #

import sigpy      as sp
import sigpy.mri  as mr
import sigpy.plot as pl

from wshfl import WaveShuffling

import mkl
mkl.set_num_threads(mkl.get_max_threads())

# -------------------------------------------------------------------------------------------------------------------------- #

rdr = np.load('data/rdr.npy')
tbl = np.load('data/tbl.npy')
mps = np.load('data/mps.npy')
psf = np.load('data/psf.npy')
phi = np.load('data/phi.npy')
mit = 300
sparse_repr = None #'W' for Wavelet; 'T' for finite differences.

# -------------------------------------------------------------------------------------------------------------------------- #

start = time.time()
Waffle = WaveShuffling(rdr, tbl, mps, psf, phi, cps=False, spr=sparse_repr, lmb=1e-6, mit=mit, dev=0)
Waffle.run()
end = time.time()
print("Reconstruction took " + str(end - start) + " seconds.")

# -------------------------------------------------------------------------------------------------------------------------- #

pl.ImagePlot(Waffle.S.H(Waffle.res).squeeze(), x=1, y=2, z=0, hide_axes=False)

# -------------------------------------------------------------------------------------------------------------------------- #
