__all__ = ['ParticleFilterFactory', 'EKFFactory']
"""
Implement filters for evaluation in main.py
"""
from warnings import warn

from numba import njit
from numpy import asarray, eye, zeros
from numpy.linalg import pinv
from numpy.random import choice


class ParticleFilterFactory:
    # region
    """Particle Filter implementation for autonomous systems, as described in
    "Probabilistic Robotics" by Sebastian Thrun. Provides indirect support for
    nonautonomous systems.

    INPUTS:
        pxt -- callable -- (X0,u,z)->X1 samples current state given prior state
        pzt -- callable -- (X1,z)->w returns probability of observations zt given state xt

        pxt_pars -- iterable, optional, default:[] -- additonal args for pxt
        pzt_pars -- iterable, optional, default:[] -- additonal args for pzt

        vec -- bool, optional, default:False -- enable vectorized pxt,pzt calls
        rbr -- bool, optional, default:False -- enable return-by-reference pxt,pzt calls
        dbg -- bool, optional, default:False -- log prediction and resampling arrays

    EXAMPLES:
        Run a single particle filter step:
        >>> pf = ParticleFilterFactory(pxt, pzt)
        >>> Xt1 = pf(Xt0, ut0, zt0)

        Run particle filter when prediction depends on timestep:
        >>> # def pxt(Xtm1, u, dt)
        >>> kwargs = {'pxt_pars': dt)
        >>> pf = ParticleFilterFactory(pxt, pzt, **kwargs)

    NOTES:
        Probability function: resampling internally normalizes the
                              probabilities, so pzt doesn't need to integrate
                              to 1.
        return-by-reference: Doesn't work as well as it should. Currently,
                             the indexing in the resampling step forces an
                             array copy anyway :( TODO

    REFERENCES:
        Thrun, Probabilistic Robotics, Chp 4.3.
        Thrun, Probabilistic Robotics, Table 4.3.
    """
    # endregion

    def __init__(self, pxt, pzt, **kwargs):
        assert callable(pxt), f'pxt must be callable, not {type(pxt)}'
        assert callable(pzt), f'pzt must be callable, not {type(pzt)}'
        self.pxt = pxt
        self.pzt = pzt
        self.pxt_pars = list(kwargs.get('pxt_pars', []))
        self.pzt_pars = list(kwargs.get('pzt_pars', []))

        self.P = kwargs.get('P',None)
        self.N =kwargs.get('N',None)

        self.dbg = kwargs.get('debug', False)
        self.Xbart_dbg = []
        self.Xt_dbg = []
        self.ii_dbg = []

        vec = kwargs.get('vec', False)
        rbr = kwargs.get('rbr', False)
        self._flow = self._flow_factory(vec, rbr)

    def __call__(self, Xtm1, ut, zt, Xt=None):
        """run particle filter
        INPUTS:
            Xtm1 -- PxN -- P particles of length-N state vectors at time t-1
            ut -- input at time t
            zt -- K -- observations at time t
            Xt -- optional, PxN -- return-by-reference output
        OUTPUTS:
            Xt -- PxN -- predicted particles at time t
        """
        Xtm1 = asarray(Xtm1)
        Xbart = self._flow(Xtm1, ut, zt)
        if self.dbg:
            self.Xbart_dbg.append(Xbart.copy())
        Xt = self._resample(Xbart)
        return Xt

    def _resample(self, Xbart):
        """resampling step"""
        P = len(Xbart)
        wt = Xbart[:, -1]
        ii = choice(range(P), size=P, p=wt / wt.sum())
        if self.dbg:
            self.ii_dbg.append(ii.copy())
        # TODO: this is advanced indexing, and so returns a copy
        return Xbart[ii, :-1]

    # ========================================================================
    # Prediction Step Factory
    #
    # We can iterate, vectorize, and return by reference
    #

    def _flow_factory(self, vec, rbr):
        if vec and rbr:
            return self._flow_vec_rbr
        elif vec:
            return self._flow_vec
        elif rbr:
            return self._flow_iter_rbr
        return self._flow_iter

    def _flow_iter(self, Xtm1, ut, zt):
        """Iterative prediction calculation. Returns directly."""
        out = zeros((Xtm1.shape[0], Xtm1.shape[1] + 1))
        return self._flow_iter_rbr(Xtm1, ut, zt, out)

    def _flow_iter_rbr(self, Xtm1, ut, zt, out):
        """Iterative prediction calculation. Returns-by-reference."""
        for i in range(len(out)):
            out[i, :-1] = self.pxt(Xtm1[i], ut, *self.pxt_pars)  # x_t^{[m]}
            out[i, -1] = self.pzt(zt, out[i, :-1], *self.pzt_pars)  # w_t^{[m]}
        return out

    def _flow_vec(self, Xtm1, ut, zt):
        """Vectorized prediction calculation. Returns directly."""
        out = zeros((Xtm1.shape[0], Xtm1.shape[1] + 1))
        return self._flow_vec_rbr(Xtm1, ut, zt, out)

    def _flow_vec_rbr(self, Xtm1, ut, zt, out):
        """Vectorized prediction calculation. Returns-by-reference."""
        out[:, :-1] = self.pxt(Xtm1, ut, *self.pxt_pars)  # x_t^{[m]}
        out[:, -1] = self.pzt(zt, out[:, :-1], *self.pzt_pars)  # w_t^{[m]}
        return out


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
    mu_t[...] = mubar + K @ (z - zhat)
    sigma_t[...] = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


@njit
def _EKF_matmuls_rbr_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t):
    """Return-by-reference EKF matmuls with njit decorator"""
    N = mubar.size
    sigmabar = G @ sigma @ G.T + R
    K = sigmabar @ H.T @ pinv(H@sigmabar@H.T + Q)
    mu_t[...] = mubar + K @ (z - zhat)
    sigma_t[...] = (eye(N) - K@H)@sigmabar
    return mu_t, sigma_t


class EKFFactory:
    # region
    """EKF implementation for autonomous systems, as described in
    "Probabilistic Robotics" by Sebastian Thrun. Provides indirect support for
    nonautonomous systems.

    Directly suppots:
        1. mixed constant and callable Jacobian, covariance matrices
        2. return-by-reference callables (g, h, matrices)
           - if used, ALL callables must return by reference
           - return-by-reference MUST be through 3rd argument
        3. inferring matrix sizes for memory preallocation
        4. return-by-reference estimation
        5. njit-optimized matrix operations
        6. additional (constant, user-supplied) parameters passed to callables

    Indirectly supports:
        1. Nonautonomous systems via direct attribute access (see Examples)
        2. Changing call signatures via subclassing (see Notes)

    REQUIRED INPUTS:
        g -- callable -- state transition function; (u,mu)->mubar
        h -- callable -- observation function; (mubar)->zhat
        G -- callable or NxN -- state transition Jacobian; (u,mu)->NxN
        H -- callable or MxN -- observation Jacobian; (mubar)->MxN
        R -- callable or NxN -- state covariance; (u,mu,sigma,z)->NxN
        Q -- callable or MxM -- observation covariance; (u,mu,sigma,z)->MxM

        Callables can also be passed additional (constant) parameters by
        means of the `*_pars` optional keyword args. See Notes.

    OPTIONAL INPUTS:
        N -- int, optional -- state space dimension, default: None
        M -- int, optional -- observation space dimention, default: None
        rbr -- bool, optional -- set true if callables return by reference, default: False
        callrbr -- bool, optional -- set true if EKF call should return by reference, default: False
        njit -- bool, optional -- use njit optimization for matrix operations, default: False
        g_pars,h_pars,H_pars,G_pars,R_pars,Q_pars -- parameters for callables. See Notes


    EXAMPLES:
        Run a single EKF step:
        >>> ekf = EKF(g,h,G,H,R,Q)
        >>> mu1, sigma1 = ekf(mu0, sigma0, u0, observation0)

        Run EKF on unforced system:
        >>> mu1, sigma1 = ekf(mu0, sigma0, 0, observation0)

        Configure to use njit-optimized matrix operations:
        >>> ekf = EKF(g,h,G,H,R,Q, njit=True)

        Configure for ALL dynamics callables returning by reference:
        >>> ekf = EKF(g,h,G,H,R,Q, rbr=True)

        Explicitly specify matrix sizes:
        >>> ekf = EKF(g,h,G,H,R,Q, N=N,M=M)

        Configure to return estimates by reference:
        >>> mu1, sigma1 = zeros_like(mu0), zeros_like(sigma0)
        >>> ekf = EKF(g,h,G,H,R,Q, callrbr=True)
        >>> ekf(mu0, sigma0, observation, mu1, sigma1)

        Run EKF when a dynamics matrix updates weirdly, e.g. over time
        >>> mu, sigma = zeros(L,N), zeros(L,N,N)
        >>> mu[-1], sigma[-1] = mu0, sigma0  # for loop compatability
        >>> ekf = EKF(g,h,G,H,R,Q)
        >>> for i in range(L):
        >>>     mu[i], sigma[i] = ekf(mu[i-1], sigma[i-1], observation[i])
        >>>     ekf.R = asarray(myWeirdRFunc(i))  # ENSURE ARRAY


    NOTES:
        N,M: Constructor will attempt to infer matrix size from matrices. This
             will not overwrite N or M if they are specified.

        return-by-reference: THIRD (3rd) function arg must be
                             return-by-reference variable. Also, ALL callables
                             must be return by reference if this option is
                             used. Since it is not possible to tell if a
                             function returns by reference, I did not provide
                             an rbr flag for each possible callable.

        Variable covariance: Covariance callables get passed (u,mu,sigma,z).
                             If you need something else, try overriding
                             the relevant wrapper call signature, e.g. _Qfun,
                             and the relevant linearization method.

        Callables with more args: Each callable also has an assosciated
                                  keyword argument, <callable_name>_pars,
                                  which is an iterable e of additional
                                  parameters to be passed when called. It will
                                  be passed like so for `g`: `g(u,mu,*g_pars)`.

        Callables with very different call signatures: Subclass and overwrite
                                                       the relevant wrappers.

    REFERENCES:
        Thrun, Probabilistic Robotics, Chp 3.3.
        Thrun, Probabilistic Robotics, Table 3.3.
    """
    # endregion

    def __init__(self, g, h, G, H, R, Q, **kwargs):
        # N=None, M=None, rbr=False, callrbr=False, njit=False):
        N, M = kwargs.get('N', None), kwargs.get('M', None)
        rbr = kwargs.get('rbr', False)
        njit = kwargs.get('njit', False)
        callrbr = kwargs.get('callrbr', False)
        for k in ('g_pars', 'h_pars', 'G_pars', 'H_pars', 'R_pars', 'Q_pars'):
            attr = kwargs.get(k, [])
            setattr(self, k, list(attr))
        N, M = self._infer_mtxsz(N, M, G, H, R, Q)
        self._init_safety_checks(g, h, N, M, rbr)

        self.G = G if callable(G) else asarray(G)
        self.H = H if callable(H) else asarray(H)
        self.R = R if callable(R) else asarray(R)
        self.Q = Q if callable(Q) else asarray(Q)

        self.mubar, self.zhat, self.G_t, self.H_t = None, None, None, None
        self.R_t, self.Q_t = None, None
        if (N is None) or (M is None):
            warn('Unable to infer matrix size. Return be reference will fail')
        else:
            self.mubar, self.zhat = zeros(N), zeros(M)
            self.G_t, self.H_t = zeros((N, N)), zeros((M, N))
            self.R_t, self.Q_t = zeros((N, N)), zeros((M, M))

        self.g, self.h, self.rbr = g, h, rbr
        self._matmuls = self._factory_matmuls(callrbr, njit)
        self._linearize = self._factory_linearize(rbr)

    def __call__(self, mu, sigma, u, z, mu_t=None, sigma_t=None):
        """run EKF - see Thrun, Probabilistic Robotics, Table 3.3 """
        mubar, zhat, G, H, R, Q = self._linearize(mu, sigma, u, z)
        return self._matmuls(G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t)

    # ========================================================================
    # Setup

    @staticmethod
    def _infer_mtxsz(N, M, G, H, R, Q):
        """Infer matrix size."""
        if N is None:
            N = len(G) if not callable(G) else N
            N = len(R) if not callable(R) else N
        if M is not None:
            M = len(Q) if not callable(Q) else M
        if not callable(H):  # H contains both
            M = len(H) if M is None else M
            N = len(H[0]) if N is None else N
        return N, M

    @staticmethod
    def _init_safety_checks(g, h, N, M, rbr):
        """Provide informative error messages to user."""
        assert callable(g), 'g must be callable'
        assert callable(h), 'h must be callable'
        if (N is None) ^ (M is None):
            warn(f'Matrix sizes only partially specified: N={N},M={M}')
        if (N is None) or (M is None):
            assert not rbr, 'cannot return-by-ref matrices of unknown size'

    # ========================================================================
    # Dynamics Wrappers
    #
    # Wrapping the dynamics objects produces cleaner code than a factory
    # approach. The user has ~5 interface choices that affect this class's
    # implementation-level code:
    #
    #   1. functions (g, h, and matrix callables)
    #       1a. return output directly
    #       1b. return by reference (i.e. have the numpy `out=` argument)
    #   2. matrices (G, H, R, Q)
    #       2a. provided as constant matrices
    #       2b. provided as callables (see above)
    #
    # This gives the class a total of ~10 possible implementations, 8 options
    # for the matrices (callable or matrix) and 2 for the callables (return
    # directly or by reference). It is easier to implement this with wrappers.
    #

    @staticmethod
    def _mtx_wrapper(obj, tgt, args):
        """Universal logic for matrix functions.
        INPUTS:
            obj -- object with desired data
            tgt -- ndarray where desired data will be stored
            args -- arguments `obj` would take if `obj` is callable
        """
        if not callable(obj):
            if tgt is None:
                return obj.view(obj.dtype)
            tgt[...] = obj.view(obj.dtype)
            return tgt
        return obj(*args)

    def _gfun(self, u, mu):
        """Wrap g with universal function call"""
        args = [u, mu, self.mubar] if self.rbr else [u, mu]
        return self.g(*args, *self.g_pars)

    def _hfun(self, mubar):
        """Wrap h with universal function call"""
        args = [mubar, self.zhat] if self.rbr else [mubar]
        return self.h(*args, *self.h_pars)

    def _Gfun(self, u, mu):
        """Wrap G with universal function call"""
        args = [u, mu, self.G_t] if self.rbr else [u, mu]
        return self._mtx_wrapper(self.G, self.G_t, args + self.G_pars)

    def _Hfun(self, mubar):
        """Wrap H with universal function call"""
        args = [mubar, self.zhat] if self.rbr else [mubar]
        return self._mtx_wrapper(self.H, self.H_t, args + self.H_pars)

    def _Rfun(self, mu, sigma, u, z):
        """Wrap R with universal function call"""
        args = [mu, sigma, u, z, self.R_t] if self.rbr else [mu, sigma, u, z]
        return self._mtx_wrapper(self.R, self.R_t, args + self.R_pars)

    def _Qfun(self, mu, sigma, u, z):
        """Wrap Q with universal function call"""
        args = [mu, sigma, u, z, self.R_t] if self.rbr else [mu, sigma, u, z]
        return self._mtx_wrapper(self.Q, self.Q_t, args + self.Q_pars)

    # ========================================================================
    # Linearization Factory
    #
    # Included more as a user courtesy. I couldn't find any noticable
    # performance improvements using return-by-reference functions, but
    # maybe someone in the future will.
    #

    def _factory_linearize(self, rbr):
        if rbr:
            return self._linearize_rbr
        return self._linearize_base

    def _linearize_base(self, mu, sigma, u, z):
        """Linearization with no interface-changing optimizations."""
        mubar = self._gfun(u, mu)
        zhat = self._hfun(mubar)
        G_t = self._Gfun(u, mu)
        H_t = self._Hfun(mubar)
        R_t = self._Rfun(mu, sigma, u, z)
        Q_t = self._Qfun(mu, sigma, u, z)
        return mubar, zhat, G_t, H_t, R_t, Q_t

    def _linearize_rbr(self, mu, sigma, u, z):
        """Linearization with return-by-reference optimizations."""
        self._gfun(u, mu)
        self._hfun(self.mubar)
        self._Gfun(u, mu)
        self._Hfun(self.mubar)
        self._Rfun(mu, sigma, u, z)
        self._Qfun(mu, sigma, u, z)
        return self.mubar, self.zhat, self.G_t, self.H_t, self.R_t, self.Q_t

    # ========================================================================
    # Matrix Multiplication Factory
    #
    # Speeding up matrix multiplication, at least via njit, provides
    # noticiable performance gains once the python bytecode is compiled.
    #

    def _factory_matmuls(self, callrbr, njit):
        if callrbr and njit:
            return self._matmuls_rbr_njit
        elif callrbr:
            return self._matmuls_rbr
        elif njit:
            return self._matmuls_njit
        return self._matmuls_base

    def _matmuls_base(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        """Basic matrix multiplication implementation."""
        return _EKF_matmuls(sigma, z, R, Q, H, G, mubar, zhat)

    def _matmuls_rbr(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        """Return-by-reference matrix multiplications."""
        return _EKF_matmuls_rbr(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t)

    def _matmuls_njit(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        """njit-optimized matrix multiplications."""
        return _EKF_matmuls_njit(sigma, z, R, Q, H, G, mubar, zhat)

    def _matmuls_rbr_njit(self, G, sigma, R, H, Q, mubar, z, zhat, mu_t, sigma_t):
        """Return-by-reference, njit-optimized matrix multiplications."""
        return _EKF_matmuls_rbr_njit(sigma, z, R, Q, H, G, mubar, zhat, mu_t, sigma_t)
