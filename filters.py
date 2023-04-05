__all__ = ['ParticleFilter']
"""
Implement filters for evaluation in main.py
"""

from numpy import asarray, einsum, eye, zeros, zeros_like
from numpy.linalg import pinv
from numpy.random import choice


class ParticleFilter():
    """Implements unforced particle filter as described in Thrun Chp4 p98.

    INPUTS:
        pxt -- callable->N -- vectorized! samples current state given prior state
        pzt -- callable->float -- vectorized! returns probability of observations
                                  zt given state xt

    USAGE:
    >>> pxt = # def p(x_t | u_t, x_{t-1})
    >>> pzt = # def p(z_t | x_t)
    >>> pf = ParticleFilter(pxt,pzt)
    >>> Xtm1 = # initialize particles
    >>> Xt = zeros((len(observations), len(Xtm1)))
    >>> for i, zt in enumerate(observations):
    >>>     Xt[i] = pf(Xtm1, zt)
    """

    def __init__(self, pxt, pzt, dbg=False):
        # sanitize
        assert callable(pxt), f'pxt must be callable, not {type(pxt)}'
        assert callable(pzt), f'pzt must be callable, not {type(pzt)}'
        # setup
        self.pxt = pxt
        self.pzt = pzt

        self.dbg = dbg
        self.Xbart_dbg = []
        self.Xt_dbg = []
        self.ii_dbg = []

    def __call__(self, Xtm1, zt):
        """run particle filter
        INPUTS:
            Xtm1 -- MxN -- M particles of length-N state vectors at time t-1
            zt -- K -- observations at time t
        OUTPUTS:
            Xt -- MxN -- resampled particles at time t
        """
        Xtm1 = asarray(Xtm1)
        Xbart = self._flow_particles(Xtm1, zt)
        if self.dbg:
            self.Xbart_dbg.append(Xbart)
        Xt = self._resample(Xbart)
        return Xt

    def _flow_particles(self, Xtm1, zt):
        out = zeros((Xtm1.shape[0], Xtm1.shape[1] + 1))
        out[:, :-1] = self.pxt(Xtm1)  # x_t^{[m]}
        out[:, -1] = self.pzt(zt, out[:, :-1])  # w_t^{[m]}
        return out

    def _resample(self, Xbart):
        """resampling step"""
        M = len(Xbart)
        wt = Xbart[:, -1]
        ii = choice(range(M), size=M, p=wt / wt.sum())
        if self.dbg:
            self.ii_dbg.append(ii)
        return Xbart[ii, :-1]


def arrmatmul(A, B, out=None):
    """Matrix multiply 2 arrays of matrices.
    INPUTS:
        A -- ...NxM -- array of NxM matrices
        B -- ...MxK -- array of MxK matrices
    OUTPUTS:
        ...NxK -- array of multiplied matrices
    """
    A, B = asarray(A), asarray(B)
    if out is None:
        out = zeros((A.shape[:-1] + (len(B.T),)))
    out = asarray(out)
    return einsum('...ij,...jk->...ik', A, B, out=out)


class ExtendedKalmanFilter():
    """unforced EKF implementation as described in Thrun Chp3 p59. """

    def __init__(self, g, h, G, H, R, Q, dbg=False):
        assert callable(g)
        assert callable(h)
        assert callable(H)
        assert callable(G)
        for k in ('g', 'h', 'G', 'R', 'H', 'Q'):
            setattr(self, k, eval(k))
        self.dbg = dbg
        self.Klog, self.Glog, self.Hlog = [], [], []

    def __call__(self, mu_t1, sigma_t1, z_t):
        """(vectorized) run EKF
        INPUTS:
            TODO
        OUTPUTS:
            TODO
        """
        mu_t1, sigma_t1, z_t = asarray(mu_t1), asarray(sigma_t1), asarray(z_t)
        mubar_t = self.g(mu_t1)
        Ht = self.H(mubar_t)
        sigmabar_t = self._calc_sigmabart(mu_t1, sigma_t1)
        K_t = self._calc_Kt(sigmabar_t, Ht)
        if self.dbg:
            Klog.append(Kt)
            Hlog.append(Ht)
            Glog.append(Gt)
        sigma_t = self._calc_sigmat(K_t, H_t, sigmabar_t)
        # TODO: check vectorization
        mu_t = mubar_t + arrmatmul(K_t, z_t - self.h(mubar_t))
        return mu_t, sigma_t

    def _calc_sigmabart(self, mu_t1, sigma_t1):
        """(vectorized) Calculate variance before measurement"""
        Gt = self.G(mu_t1)
        sigmabar_t = arrmatmul(Gt, sigma_t1)
        sigmabar_t = arrmatmul(sigmabar_t, Gt)
        sigmabar_t += self.R  # TODO: check vectorization
        return sigmabar_t

    def _calc_Kt(self, sigmabar_t, Ht):
        """(vectorized) Calculate Kalman Gain"""
        H_tT = H_t.transpose(*H_t.shape[:-2], H_t.shape[-1], H_t.shape[-2])
        sigmaHT = arrmatmul(sigmabar_t, H_tT)
        HsigmaHT = arrmatmul(H_t, sigmaHT)
        HsigmaHT_plus_Q = HsigmaHT + self.Q  # TODO: check vectorization
        return arrmatmul(sigmaHT, pinv(HsigmaHT_plus_Q))

    def _calc_sigmat(self, K_t, H_t, sigmabar_t):
        """(vectorized) Calculate variance after measurement"""
        KH = arrmatmul(Kt, Ht)
        I = zeros_like(KH)
        I[:] = eye(len(KH.T))
        ImKH = I - KH  # TODO: check vectorization
        return arrmatmul(ImKH, sigmabar_t)
