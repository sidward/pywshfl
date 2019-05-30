#!/usr/bin/env python3

import numpy      as np
import sigpy      as sp
import sigpy.mri  as mr
import sigpy.plot as pl

# Look at:
# sigpy.alg.GradientMethod

class WaveShuffling(sp.app.App):

  def _construct_kernel(self):

    self.kernel = self.xp.zeros([1, self.sy, self.sz, 1, 1, 1, self.tk, self.tk]).astype(self.xp.complex64)
    vec = self.xp.zeros((self.tk, 1)).astype(self.xp.complex64)
    mask = self.xp.zeros([self.sy, self.sz, self.tf]).astype(self.xp.complex64)

    phi = self.phi.squeeze()
    if (len(phi.shape) == 1):
      phi = self.xp.reshape(phi, (phi.size, 1))
    U = sp.linop.MatMul(vec.shape, phi)

    for k in range(self.rdr.shape[0]):
      [ky, kz, ec] = self.rdr[k, :]
      mask[ky, kz, ec] = 1

    flags = set() 
    for k in range(self.rdr.shape[0]):
      [ky, kz, ec] = self.rdr[k, :]
      key = str(ky) + ',' + str(kz)
      if key in flags:
        continue
      flags.add(key)
      mvec = mask[ky, kz, :].squeeze()

      for t in range(self.tk):
        vec = vec * 0
        vec[t] = 1
        self.kernel[0, ky, kz, 0, 0, 0, t, :] = U.H(U(vec).squeeze() * mvec).squeeze()

  def _construct_AHb(self):
    x = self.xp.zeros((self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1)).astype(self.xp.complex64)
    PhiH = self.phi.squeeze().conj()
    if (len(PhiH.shape) == 1):
      PhiH = self.xp.reshape(PhiH, (PhiH.size, 1))
    for k in range(self.rdr.shape[0]):
      [ky, kz, ec] = self.rdr[k, :]
      for t in range(self.tk):
        x[:, ky, kz, :, 0, 0, t, 0] = self.tbl[:, :, k] * PhiH[ec, t]
    self.AHb = self.W(self.E.H(self.R.H(self.Fx.H(self.Psf.H(self.Fyz.H(x))))))

  def _broadcast_check(self, x):
    if(len(x.shape) == self.max_dims):
      return x
    x = sp.to_device(x, self.cpu)
    while (len(x.shape) < self.max_dims):
      x = np.expand_dims(x, axis=(len(x.shape) + 1))
    return x

  def __init__(self, rdr, tbl, mps, psf, phi, cps, lmb, mit, dev):

    self.cpu = -1
    self.max_dims = 8
    device = sp.Device(dev)
    self.xp = device.xp

    with device:
      self.rdr = self.xp.array(rdr).astype(self.xp.int32)
      self.tbl = self.xp.array(tbl)
      self.mps = self.xp.array(self._broadcast_check(mps))
      self.psf = self.xp.array(self._broadcast_check(psf))
      self.phi = self.xp.array(self._broadcast_check(phi))
      self.cps = cps # Caipi-Shuffling flag.
      self.lmb = lmb # Lambda.
      self.mit = mit # Max-Iter

      self.wx = self.psf.shape[0]
      self.sx = self.mps.shape[0]
      self.sy = self.mps.shape[1]
      self.sz = self.mps.shape[2]
      self.nc = self.mps.shape[3]
      self.md = self.mps.shape[4]
      self.tf = self.phi.shape[5]
      self.tk = self.phi.shape[6]

      wavelet_axes = tuple([k for k in range(3) if self.mps.shape[k] > 1])

      assert(self.md == 1) # Until multiple ESPIRiT maps is implemented.

      self._construct_kernel()

      self.E      = sp.linop.Multiply([self.sx, self.sy, self.sz, 1, self.md, 1, self.tk, 1], self.mps)
      self.R      = sp.linop.Resize([self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1], \
                                  [self.sx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1])
      self.Fx     = sp.linop.FFT([self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1], axes=(0,))
      self.Psf    = sp.linop.Multiply([self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1], self.psf)
      self.Fyz    = sp.linop.FFT([self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1], axes=(1, 2))
      self.K      = sp.linop.Sum(     [self.wx, self.sy, self.sz, 1, 1, 1, self.tk, self.tk], axes=(6,)) * \
                    sp.linop.Multiply([self.wx, self.sy, self.sz, 1, 1, 1, self.tk,       1], self.kernel)
      self.W      = sp.linop.Wavelet([self.sx, self.sy, self.sz, 1, 1, 1, self.tk, 1], axes=wavelet_axes)

      self._construct_AHb()

      # Only need to define grad f and prox g
      #self.AHA = E
      #proxg = sp.prox.L1Reg(A.ishape, lamda)
        
      #self.wav = np.zeros(A.ishape, np.complex)
      #alpha = 1

      #def gradf(x):
      # return A.H * (A * x - ksp)

      #alg = sp.alg.GradientMethod(gradf, self.wav, alpha, proxg=proxg, max_iter=max_iter)
      #super().__init__(alg)
        
    #def _output(self):
    #    return self.W.H(self.wav)

#    def _summarize():
#    def objective();
