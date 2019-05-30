import warnings
#warnings.filterwarnings('ignore')
import os
import sys 
path = os.environ["TOOLBOX_PATH"] + "/python/";
sys.path.append(path);
import numpy as np
from bart import bart

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
cps = False
lmb = 0
mit = 1 
dev = -1
alp = 10

# ----------------------------------------------------------------------------------------------------------------------------- #

res = bart(1, 'wshfl -r 1e-5 -i 300 -t 1e-3 -f -H -w', np.transpose(mps), np.transpose(psf), np.transpose(phi), rdr, tbl);

# ----------------------------------------------------------------------------------------------------------------------------- #

Waffle = WaveShuffling(rdr, tbl, mps, psf, phi, cps=False, lmb=1e-5, mit=mit, alp=1, dev=-1, tol=1e-3)
Waffle.run()

# ----------------------------------------------------------------------------------------------------------------------------- #
