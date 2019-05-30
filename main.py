#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------------------------- #

import warnings
#warnings.filterwarnings('ignore')
import os
import sys 
path = os.environ["TOOLBOX_PATH"] + "/python/";
sys.path.append(path);
import numpy as np
from bart import bart
import matplotlib.pyplot as plt
import time


# ----------------------------------------------------------------------------------------------------------------------------- #

import sigpy      as sp
import sigpy.mri  as mr
import sigpy.plot as pl

from wshfl import WaveShuffling

# ----------------------------------------------------------------------------------------------------------------------------- #

rdr = np.load('data/rdr.npy')
tbl = np.load('data/tbl.npy')
mps = np.load('data/mps.npy')
psf = np.load('data/psf.npy')
phi = np.load('data/phi.npy')
mit = 300

# ----------------------------------------------------------------------------------------------------------------------------- #

start = time.time()
res_1 = bart(1, 'wshfl -r 1e-6 -i %d -t 1e-3 -f -H -w' % mit, np.transpose(mps), np.transpose(psf), np.transpose(phi), rdr, tbl)
end = time.time()
print("BART took " + str(end - start) + " seconds.")
res_1 = res_1.squeeze()

# ----------------------------------------------------------------------------------------------------------------------------- #

start = time.time()
Waffle = WaveShuffling(rdr, tbl, mps, psf, phi, lmb=1e-6, mit=mit, dev=0)
Waffle.run()
end = time.time()
print("SIGPY took " + str(end - start) + " seconds.")
res_2 = np.transpose(Waffle.W.H(sp.to_device(Waffle.res, -1)).squeeze())

# ----------------------------------------------------------------------------------------------------------------------------- #

res_1 = res_1/np.max(res_1)
res_2 = res_2/np.max(res_2)

img_1 = np.concatenate((res_1[:, :, 0], res_2[:, :, 0]), axis=1)
img_2 = np.concatenate((res_1[:, :, 1], res_2[:, :, 1]), axis=1)
img   = np.concatenate((img_1, img_2), axis=0)

plt.imshow(np.abs(img))
plt.show()
