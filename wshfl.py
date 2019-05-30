#!/usr/bin/env python3

import sigpy      as sp
import sigpy.mri  as mr
import sigpy.plot as plt

# Look at:
# sigpy.alg.GradientMethod

class WshflWaveletRecon(sp.app.App):

  def _construct_kernel(reorder, phi):
    # TODO

  def __init__(self, rdr, tbl, mps, wve, phi, cps, lmb, mit):

    self.rdr = rdr
    self.tbl = tbl 
    self.mps = mps 
    self.wve = wve
    self.phi = phi
    self.cps = cps # Caipi-Shuffling
    self.lmb = lmb # Lambda.
    self.mit = mit # Max-Iter

    self.wx = wve.shape[0]
    self.sx = mps.shape[0]
    self.sy = mps.shape[1]
    self.sz = mps.shape[2]
    self.nc = mps.shape[3]
    self.md = mps.shape[4]
    self.tf = phi.shape[5]
    self.tk = phi.shape[6]

    # Only need to define grad f and prox g
        
    self.E   = sp.linop.Multiply([self.sx, self.sy, self.sz, 1, 1, 1, self.tk], self.mps)
    self.Fx  = sp.linop.FFT(ksp.shape, axes=(0,))
    self.W   = sp.linop.Multiply([self.sx, self.sy, self.sz, 1, 1, 1, 1], self.wve)
    self.Fyz = sp.linop.FFT(ksp.shape, axes=(1, 2))

    P = sp.linop.Multiply(ksp.shape, mask)
        self.W = sp.linop.Wavelet(img_shape)
        A = P * F * S * self.W.H
        
        proxg = sp.prox.L1Reg(A.ishape, lamda)
        
        self.wav = np.zeros(A.ishape, np.complex)
        alpha = 1

        def gradf(x):
            return A.H * (A * x - ksp)

        alg = sp.alg.GradientMethod(gradf, self.wav, alpha, proxg=proxg, 
                                    max_iter=max_iter)
        super().__init__(alg)
        
    def _output(self):
        return self.W.H(self.wav)

#    def _summarize():
#    def objective();


print("Test")
