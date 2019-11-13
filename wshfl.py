#!/usr/bin/env python3

import numpy      as np
import sigpy      as sp
import sigpy.mri  as mr
import sigpy.plot as pl

class WaveShuffling(sp.app.App):

  def _construct_AHb(self):
    x = self.xp.zeros((1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx)).astype(self.xp.complex64)
    PhiH = self.phi.squeeze().conj()
    if (len(PhiH.shape) == 1):
      PhiH = self.xp.reshape(PhiH, (1, PhiH.size))
    for k in range(self.rdr.shape[0]):
      [ky, kz, ec] = self.rdr[k, :]
      for t in range(self.tk):
        x[0, t, 0, 0, :, kz, ky, :] = (self.tbl[:, :, k] * PhiH[t, ec]).T
    self.AHb = self.S(self.E.H(self.R.H(self.Fx.H(self.Psf.H(self.Fyz.H(x))))))

  def _construct_kernel(self):
    self.kernel = self.xp.zeros([self.tk, self.tk, 1, 1, 1, self.sz, self.sy, 1]).astype(self.xp.complex64)
    vec = self.xp.zeros((self.tk, 1)).astype(self.xp.complex64)
    mask = self.xp.zeros([self.sy, self.sz, self.tf]).astype(self.xp.complex64)

    phi = self.phi.squeeze().T
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
        self.kernel[:, t, 0, 0, 0, kz, ky, 0] = U.H(U(vec).squeeze() * mvec).squeeze()

  def _broadcast_check(self, x):
    if(len(x.shape) == self.max_dims):
      return x
    x = sp.to_device(x, self.cpu)
    while (len(x.shape) < self.max_dims):
      x = np.expand_dims(x, axis=0)
    return x

  def __init__(self, rdr, tbl, mps, psf, phi, spr='W', cft=True, \
                  lmb=1e-5, mit=30, alp=0.25, tol=1e-3, dev=-1):
    self.cpu = -1
    self.max_dims = 8
    self.device = sp.Device(dev)
    self.xp = self.device.xp
    self.center = cft

    with self.device:
      self.rdr = self.xp.array(rdr).astype(self.xp.int32)
      self.tbl = self.xp.array(tbl)
      self.tbl = self.tbl/self.xp.max(self.xp.abs(self.tbl))
      self.mps = self.xp.array(self._broadcast_check(mps))
      self.psf = self.xp.array(self._broadcast_check(psf))
      self.phi = self.xp.array(self._broadcast_check(phi))
      self.lmb = lmb # Lambda.
      self.mit = mit # Max-Iter
      self.alp = alp # Step size.
      self.tol = tol # Tolerance.

      self.wx = self.psf.shape[7]
      self.sx = self.mps.shape[7]
      self.sy = self.mps.shape[6]
      self.sz = self.mps.shape[5]
      self.nc = self.mps.shape[4]
      self.md = self.mps.shape[3]
      self.tf = self.phi.shape[2]
      self.tk = self.phi.shape[1]

      assert(self.md == 1) # Until multiple ESPIRiT maps is implemented.

      self.S = None
      if (spr == 'W'):
        wavelet_axes = tuple([k for k in range(5, 8) if self.mps.shape[k] > 1])
        self.S = sp.linop.Wavelet([1, self.tk, 1, self.md, 1, self.sz, self.sy, self.sx], axes=wavelet_axes)
      elif (spr == 'T'):
        self.S = sp.linop.FiniteDifference([1, self.tk, 1, self.md, 1, self.sz, self.sy, self.sx], axes=(5, 6, 7))
      else:
        self.S = sp.linop.Identity([1, self.tk, 1, self.md, 1, self.sz, self.sy, self.sx])

      self.E      = sp.linop.Multiply([1, self.tk, 1, self.md, 1, self.sz, self.sy, self.sx], self.mps)
      self.R      = sp.linop.Resize([1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx], \
                                    [1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.sx])
      self.Fx     = sp.linop.FFT([1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx], axes=(7,), \
                      center=self.center)
      self.Psf    = sp.linop.Multiply([1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx], self.psf)
      self.Fyz    = sp.linop.FFT([1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx], axes=(5, 6), \
                      center=self.center)

      self.K = None
      self._construct_kernel()
      self.K    = sp.linop.Reshape( [      1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx],              \
                                      [         self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx])            * \
                  sp.linop.Sum(     [self.tk, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx], axes=(1,)) * \
                  sp.linop.Multiply([      1, self.tk, 1, 1, self.nc, self.sz, self.sy, self.wx], self.kernel)

      self._construct_AHb()
      self.res   = 0 * self.AHb
      self.AHA   = self.S * self.E.H * self.R.H * self.Fx.H * self.Psf.H * self.Fyz.H * self.K * \
                   self.Fyz * self.Psf * self.Fx * self.R * self.E * self.S.H
      self.mxevl = sp.app.MaxEig(self.AHA, self.xp.complex64, device=dev).run()
      self.alp   = self.alp/self.mxevl
      self.gradf = lambda x: self.AHA(x) - self.AHb
      self.proxg = sp.prox.L1Reg(self.res.shape, lmb)

      alg = sp.alg.GradientMethod(self.gradf, self.res, self.alp, proxg=self.proxg, accelerate=True, tol=self.tol, max_iter=self.mit)
      super().__init__(alg)
        
    def _output(self):
        return self.S.H(self.res)
