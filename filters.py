__all__ = ['ParticleFilter', 'ExtendedKalmanFilter']
"""
Implement filters for evaluation in main.py
"""
from warnings import warn

from numba import jit, njit
from numpy import asarray, dot, eye, matmul, newaxis, zeros, zeros_like
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


def _EKF_matmuls(sigma, z, R, Q, H, G, mubar, zhat):
    """Generic EKF matmuls"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t = mubar + K @ (z - zhat)
    sigma_t = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


@njit
def _EKF_matmuls_njit(sigma, z, R, Q, H, G, mubar, zhat):
    """Generic EKF matmuls with njit decorator"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t = mubar + K @ (z - zhat)
    sigma_t = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


def _EKF_matmuls_rbr(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t):
    """Return-by-reference EKF matmuls"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t = mubar + K @ (z - zhat)
    sigma_t = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


@njit
def _EKF_matmuls_rbr_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t):
    """Return-by-reference EKF matmuls with njit decorator"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t = mubar + K @ (z - zhat)
    sigma_t = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t

# @njit


def _EKF_matmuls_prealloc(sigma, z, R, Q, H, G, mubar, hmubar, mu_t, sigma_t,
                          sigmabar, K, I, HsHT):
    """EKF matrix muls optimized for no memory allocation

    Requires preallocation of matrices
        - sigmabar, K, I, hmubar

    https://github.com/numba/numba/issues/3804
    """
    # Sigma bar
    # ---------
    dot(G, sigma, out=sigmabar)
    dot(sigmabar, G.T, out=sigmabar)
    sigmabar += R
    # Kalman Gain
    # -----------
    dot(sigmabar, H.T, out=K)
    dot(H, K, out=HsHT)
    HsHT += Q
    HsHT[...] = pinv(HsHT)
    dot(K, HsHT, out=K)
    # mu
    # --
    dot(K, z - hmubar, out=mu_t)
    mu_t += mubar
    # sigma
    # -----
    dot(-K, H, out=sigma_t)
    sigma_t += I
    dot(sigma_t, sigmabar, out=sigma_t)

    return mu_t, sigma_t


class ExtendedKalmanFilter():
    # region
    """EKF implementation for autonomous systems
    TODO: pre-allocate matrices where possible
    TODO: njit option

    INPUTS:
        g -- callable -- state transition (u,mu)->mubar
        h -- callable -- observation (mubar)->zhat
        G -- callable or NxN -- state transition Jacobian (u,mu)->NxN
        H -- callable or MxN -- observation Jacobian (mubar)->MxN
        R -- callable or NxN -- state covariance (u,mu,sigma,z)->NxN
        Q -- callable or MxM -- observation covariance (u,mu,sigma,z)->MxM
        N -- int, optional -- state space dimenstion, default: None
        M -- int, optional -- observation space dimenstion, default: None
        rbr -- bool, optional -- set true if callables return by reference, default: False
    EXAMPLE:

    NOTES:
        N,M: Constructor will attempt to infer matrix size from matrices. This
             will not overwrite N or M if they are specified.
        R,Q: - as funcs of time
             - APT descr
    REFERENCES:
      Thrun, Probabilistic Robotics, Chp 3.3.
      Thrun, Probabilistic Robotics, Table 3.3.
    """
    # endregion

    def __init__(self, g, h, G, H, R, Q, N=None, M=None, rbr=False, njit=False):
        assert callable(g), 'g must be callable'
        assert callable(h), 'h must be callable'
        self.g, self.h, self.G, self.H, self.R, self.Q = g, h, G, H, R, Q
        for k in ('G', 'H', 'R', 'Q'):
            attr = getattr(self, k)
            if not callable(attr):
                setattr(self, k, asarray(attr))

        N, M = self._infer_mtxsz(N, M)
        if (N is not None) and (M is not None):
            self._prealloc(N, M)
        elif (N is None) ^ (M is None):
            warn(f'Matrix sizes only partially specified: N={N},M={M}')
            assert not rbr, 'cannot return-by-ref matrices of unknown size'

        self.rbr = rbr
        self._matmuls = self._factory_matmuls(njit)

    def __call__(self, mu, sigma, u, z, mu_t=None, sigma_t=None):
        """run EKF
        INPUTS:
            TODO
        OUTPUTS:
            TODO
        """
        # if mu_t is None:
        #     mu_t = zeros_like(mu)
        # if sigma_t is None:
        #     sigma_t = zeros_like(sigma)
        mubar, zhat, G, H, R, Q = self._linearize(mu, sigma, u, z)
        return self._matmuls(G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t)

    def _linearize(self, mu, sigma, u, z):
        """linearization calcs with no explicity optimizations"""
        mubar = self.gfun(u, mu)
        zhat = self.hfun(mubar)
        G_t = self.Gfun(u, mu)
        H_t = self.Hfun(mubar)
        R_t = self.Rfun(mu, sigma, u, z)
        Q_t = self.Qfun(mu, sigma, u, z)
        return mubar, zhat, G_t, H_t, R_t, Q_t

    # ========================================================================
    # Factory Methods

    def _infer_mtxsz(self, N, M):
        """Infer matrix size. Requires G,H,R,Q attributes defined"""
        if N is None:  # infer N
            N = len(self.G) if not callable(self.G) else N
            N = len(self.R) if not callable(self.R) else N
        if M is not None:  # infer M
            M = len(self.Q) if not callable(self.Q) else M
        # H contains both
        if not callable(self.H):
            M = len(self.H) if M is None else M
            N = len(self.H.T) if N is None else N
        return N, M

    def _prealloc(self, N, M):
        # TODO: return explicitly
        self.mubar = zeros(N)
        self.zhat = zeros(M)
        self.G_t = eye(N)
        self.H_t = zeros((M, N))
        self.R_t = eye(N)
        self.Q_t = eye(M)
        # TODO: I, sigmabar, K, HsHT

    # ========================================================================
    # Wrappers
    # TODO: would factory pattern boost speed?

    def gfun(self, u, mu):
        args = [u, mu]
        if self.rbr:
            args += [self.mubar]
        return self.g(*args)

    def hfun(self, mubar):
        args = [mubar]
        if self.rbr:
            args += [self.zhat]
        return self.h(*args)

    def Gfun(self, u, mu):
        if not callable(self.G):
            return self.G
        args = [u, mu]
        if self.rbr:
            args += [self.G_t]
        return self.G(*args)

    def Hfun(self, mubar):
        if not callable(self.H):
            return self.H
        args = [mubar]
        if self.rbr:
            args += [self.zhat]
        return self.G(*args)

    def Rfun(self, mu, sigma, u, z):
        if not callable(self.R):
            return self.R
        args = [mu, sigma, u, z]
        if self.rbr:
            args += [self.R_t]
        return self.R(*args)

    def Qfun(self, mu, sigma, u, z):
        if not callable(self.Q):
            return self.Q
        args = [mu, sigma, u, z]
        if self.rbr:
            args += [self.Q_t]
        return self.Q(*args)

    # ========================================================================
    # Linearization Methods

    # def _linearize_rbr(self, mu, u):
    #     """linearization calcs with return-by-reference optimizations"""
    #     self.gfun(u, mu, self.mubar)
    #     self.hfun(self.mubar, self.zhat)
    #     self.Gfun(u, mu, self.G_t)
    #     self.Hfun(self.mubar, self.H_t)
    #     # self.Rfun(, self.R_t)
    #     # self.Qfun(, self.Q_t)
    #     return self.mubar, self.zhat, self.G_t, self.H_t, self.R_t, self.Q_t

    # ========================================================================
    # MatMul Methods

    def _factory_matmuls(self, njit):
        if self.rbr and njit:
            return self._matmuls_rbr_njit
        elif self.rbr:
            return self._matmuls_rbr
        elif njit:
            return self._matmuls_njit
        else:
            return self._matmuls_base

    def _matmuls_base(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        return _EKF_matmuls(sigma, z, R, Q, H, G, mubar, zhat)

    def _matmuls_rbr(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        return _EKF_matmuls_rbr(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t)

    def _matmuls_njit(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        return _EKF_matmuls_njit(sigma, z, R, Q, H, G, mubar, zhat)

    def _matmuls_rbr_njit(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        return _EKF_matmuls_rbr_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t)
