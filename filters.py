__all__ = ['ParticleFilter']
"""
Implement filters for evaluation in main.py
"""

from numpy import asarray, zeros, zeros_like
from numpy.random import choice


class ParticleFilter():
    """Implements particle filter as described in Thrun Chp4 p98. 

    INPUTS:
        pxt -- callable->N -- samples current state given prior state
        pzt -- callable->float -- returns probability of observations zt
                                  given state xt

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
        for m in range(len(Xtm1)):
            xt = self.pxt(Xtm1[m])  # x_t^{[m]}
            wt = self.pzt(zt, xt)  # w_t^{[m]}
            out[m, :-1] = xt
            out[:, -1] = wt
        return out

    def _resample(self, Xbart):
        """resampling step"""
        M = len(Xbart)
        wt = Xbart[:, -1]
        ii = choice(range(M), size=M, p=wt / wt.sum())
        if self.dbg:
            self.ii_dbg.append(ii)
        return Xbart[ii, :-1]

