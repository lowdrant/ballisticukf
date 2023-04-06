__all__ = ['ParticleFilter', 'ExtendedKalmanFilter']
"""
Implement filters for evaluation in main.py
"""
from numba import njit, jit
from numpy import asarray, eye, matmul, newaxis, zeros, zeros_like, dot
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


@njit
def _EKF_matmuls(sigma, z, R, Q, H, G, mubar, hmubar, mu_t, sigma_t):
    """Generic EKF matmuls"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t = mubar + K @ (z - hmubar)
    sigma_t = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


@njit
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
    """EKF implementation as described in Thrun Chp3 p59.

    TODO: pre-allocate matrices where possible
    TODO: debuggers
    TODO: API for R,Q matrices
    """

    def __init__(self, g, h, G, H, R, Q, N=None, M=None, pbr=False, dbg=False):
        self.g, self.h = g, h
        assert callable(self.g), 'g must be callable'
        assert callable(self.h), 'h must be callable'
        self.N, self.M = N, M
        self.G, self.H, self.R, self.Q = G, H, R, Q
        self.pbr = pbr
        # can_prealloc = self._determine_prealloc()
        #self._filterfun = self._call_factory()
        self._filterfun = self._filter_static

    # ========================================================================
    # Call Setup

    def _determine_prealloc(self):
        if self.N is not None and self.M is not None:
            return True

        for k in ('G', 'H', 'R', 'Q'):
            attr = getattr(self, k)
            if callable(attr):
                anycall = True
                nonecall = False
        assert anycall ^ nonecall, 'issue determining callable matrices'
        allcall = anycall and not nonecall

    def _call_factory(self, can_prealloc):
        """Go over EKF matrix params and determine if callable or not
        and adjust class params accordingly
        """
        if nonecall:
            return self._filter_static
    # '''

    def __call__(self, mu, sigma, u, z, mu_t=None, sigma_t=None):
        """run EKF
        INPUTS:
            TODO
        OUTPUTS:
            TODO
        """
        if mu_t is None:
            mu_t = zeros_like(mu)
        if sigma_t is None:
            sigma_t = zeros_like(sigma)
        return self._filterfun(mu, sigma, u, z, mu_t, sigma_t)

    # ========================================================================
    # Interface Wrappers
    def gwrap(self, u, mu):
        if self.pbr:
            return self.g(u, mu, self.mubar)
        return self.g(u, mu)

    def hwrap(self, mubar):
        if self.pbr:
            return self.h(mubar, self.h_mubar)
        return self.h(mubar)

    # ========================================================================
    # Basic Filter Implementations

    def _filter_base(self, mu, sigma, u, z, mu_t, sigma_t):
        """Generic EKF implementation"""
        mubar = self.g(u, mu)
        H, G = self.H(mubar), self.G(u, mu)
        R = self.R(mu, sigma, u, z)
        Q = self.Q(mu, sigma, u, z)
        hmubar = self.h(mubar)
        return _EKF_matmuls(sigma, z, R, Q, H, G, mubar, hmubar, mu_t, sigma_t)

    def _filter_static(self, mu, sigma, u, z, mu_t, sigma_t):
        """EKF with constant matrices"""
        mubar = self.g(u, mu)
        hmubar = self.h(mubar)
        N, M = len(mubar), len(hmubar)
        sigmabar = zeros((N, N))
        K = zeros((N, M))
        I = eye(N)
        HsHT = zeros((M, M))
        # return _EKF_matmuls(sigma, z, self.R, self.Q, self.H,
        #                              self.G, mubar, hmubar,mu_t, sigma_t)
        return _EKF_matmuls_prealloc(sigma, z, self.R, self.Q, self.H,
                                     self.G, mubar, hmubar, mu_t, sigma_t,
                                     sigmabar, K, I, HsHT)

    # ========================================================================
    # Memory Pre-Alloc Methods

    def _prealloc(self):
        """allocate intermediate arrays once for efficiency/njit
        compatability"""
        self.I = eye(self.N)
        self.mubar = zeros(self.N)
        self.h_mubar = zeros(self.M)
        self.sigmabar = zeros((self.N, self.N))
        self.K = zeros((self.N, self.M))
        self.Rt = zeros_like(self.sigmabar)
        self.Qt = zeros((self.M, self.M))

    def _filter_prealloc(self, mu, sigma, u, z, mu_t, sigma_t):
        self.g(u, mu, self.mubar)
        self.h(mubar, self.h_mubar)
        return _EKF_matmuls_prealloc(self, sigma, z, R, Q, H, G, self.mubar, mu_t, sigma_t,
                                     self.sigmabar, self.K, self.I, self.hmubar)
