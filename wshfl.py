#!/usr/bin/env python3

import numpy      as np
import sigpy      as sp
import sigpy.mri  as mr
import sigpy.plot as pl

# Look at:
# sigpy.alg.GradientMethod

class WaveShuffling(sp.app.App):

  def _construct_kernel(self):

    self.kernel = self.xp.zeros([1, self.sy, self.sz, 1, 1, 1, self.tk, self.tk])
    vec = self.xp.zeros((self.tk, 1))
    mask = self.xp.zeros([1, self.sy, self.sz, 1, 1, self.tf, 1, 1])

    U = sp.linop.MatMul(vec.shape, self.phi.squeeze())

    for k in range(self.rdr.shape[0]):
      [ky, kz, ec] = self.rdr[k, :]
      mask[0, ky, kz, 0, 0, ec, 0, 0] = 1

    flags = set() 
    for k in range(self.rdr.shape[0]):
      [ky, kz, ec] = self.rdr[k, :]
      key = str(ky) + ',' + str(kz)
      if key in flags:
        continue
      flags.add(key)
      mvec = mask[0, ky, kz, 0, 0, :, 0, 0].squeeze()

      for t in range(self.tk):
        vec = vec * 0
        vec[t] = 1
        self.kernel[0, ky, kz, 0, 0, 0, t, :] = U.H(U(vec).squeeze() * mvec).squeeze()

  #def _construct_AHb(rdr, tbl, phi):

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
      self.cps = cps # Caipi-Shuffling
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

      self._construct_kernel()
      print(self.kernel.shape)

      self.E   = sp.linop.Multiply([self.sx, self.sy, self.sz, 1, 1, 1, self.tk, 1], self.mps)
      self.R   = sp.linop.Resize([self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1], \
                                 [self.sx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1])
      self.Fx  = sp.linop.FFT([self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1], axes=(0,))
      self.Psf = sp.linop.Multiply([self.sx, self.sy, self.sz, self.nc, 1, 1, 1], self.psf)
      self.Fyz = sp.linop.FFT([self.wx, self.sy, self.sz, self.nc, 1, 1, self.tk, 1], axes=(1, 2))
      self.K   = sp.linop.MatMul([self.wx, self.sy, self.sz, 1, 1, 1, self.tk, 1], self.kernel)
      self.Wav = sp.linop.Wavelet([self.wx, self.sy, self.sz, 1, 1, 1, self.tk, 1], axes=(0, 1, 2))

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
