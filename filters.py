__all__ = ['ParticleFilter', 'ExtendedKalmanFilter']
"""
Implement filters for evaluation in main.py
"""

from numpy import asarray, einsum, eye, newaxis, zeros, zeros_like
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
    """unforced EKF implementation as described in Thrun Chp3 p59.

    TODO: __call__ factory in init 
        - allow callable or matrices for G,H,R,Q
    TODO: pre-allocate matrices where possible
    TODO: remove vectorization since only 1 "particle" per call
    """

    def __init__(self, g, h, G, H, R, Q, dbg=False):
        for k in ('g', 'h', 'G', 'H'):
            fun = eval(k)
            assert callable(fun), f'{k} must be callable'
            setattr(self, k, fun)
        self.R, self.Q = asarray(R), asarray(Q)
        self.N, self.K = len(self.R), len(self.Q)

        # debuggers
        self.dbg = dbg
        self.Klog, self.Glog, self.Hlog = [], [], []
        # if dbg:
        #    self.__call__ = self._call_dbg
        #self.__call__ = self._call_base

    def __call__(self, mu_t1, sigma_t1, z_t):
        """(vectorized) run EKF
        INPUTS:
            TODO
        OUTPUTS:
            TODO
        """
        mu_t1, sigma_t1, z_t = asarray(mu_t1), asarray(sigma_t1), asarray(z_t)
        mubar_t = self.g(mu_t1)
        H_t, G_t = self.H(mubar_t), self.G(mu_t1)
        sigmabar_t = self._calc_sigmabart(G_t, mu_t1, sigma_t1, self.R)
        K_t = self._calc_Kt(sigmabar_t, H_t, self.Q)
        sigma_t = self._calc_sigmat(K_t, H_t, sigmabar_t)
        mu_t = (z_t - self.h(mubar_t).squeeze()).T
        mu_t = K_t @ mu_t
        mu_t += mubar_t.T
        return mu_t, sigma_t

    def _calc_sigmabart(self, G_t, mu_t1, sigma_t1, R_t, out=None):
        out = G_t @ sigma_t1 @ G_t.T + R_t
        return out

    def _calc_Kt(self, sigmabar_t, H_t, Q_t, out=None):
        out = sigmabar_t @ H_t.T @ pinv(H_t @ sigmabar_t @ H_t.T + Q_t)
        return out

    def _calc_sigmat(self, K_t, H_t, sigmabar_t):
        KH = K_t @ H_t
        I = zeros_like(KH)
        I[:] = eye(len(KH.T))
        ImKH = I - KH
        return ImKH @ sigmabar_t

    def _prealloc(self):
        """allocate intermediate arrays once for efficiency/njit
        compatability"""
        raise NotImplementedError
        self.I = eye(self.K)
