#!/usr/bin/env python3

# -------------------------------------------------------------------------------------------------------------------------- #

import time
import numpy as np

# -------------------------------------------------------------------------------------------------------------------------- #

import sigpy      as sp
import sigpy.mri  as mr
import sigpy.plot as pl

from wshfl import WaveShuffling

# -------------------------------------------------------------------------------------------------------------------------- #

rdr = np.load('data/rdr.npy')
tbl = np.load('data/tbl.npy')
mps = np.load('data/mps.npy').T
psf = np.load('data/psf.npy').T
phi = np.load('data/phi.npy').T
mit = 300
sparse_repr = 'W'

# -------------------------------------------------------------------------------------------------------------------------- #

start = time.time()
Waffle = WaveShuffling(rdr, tbl, mps, psf, phi, spr=sparse_repr, lmb=1e-6, mit=mit, dev=0)
Waffle.run()
end = time.time()
print("Device used: " + str(Waffle.device) + ". Reconstruction took " + str(end - start) + " seconds.")

# -------------------------------------------------------------------------------------------------------------------------- #

pl.ImagePlot(Waffle.S.H(Waffle.res).squeeze(), x=1, y=2, z=0, hide_axes=False)

# -------------------------------------------------------------------------------------------------------------------------- #
