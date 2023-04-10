__all__ = ['pxt', 'pzt', 'construct_pf']
from itertools import combinations
from time import time

from numpy import *
from numpy.random import normal, seed

from filters import ParticleFilterFactory
from helpers import *

# ===================================================================
# Priors


def pxt_rbr(xtm1, u, out, dt, scale):
    """generate probability cloud of current state given prev state
    WARN: for relative marker positions
    INPUTS:
        xtm1 -- PxN -- P prior estimates for N states
                       Format: (x,y, vx,vy, w, mx0,my0,...,mxK,myK)
    NOTES:
        Modeling motion of off-axis point, since that's what we are observing

        For a disk at (x,y) moving at (vx,vy) with angle a and angular
        velocity w, a point at radius r will have
        observed velocity vo (vx+w*r*sina, vy+w*r*cosa)
    """
    loc = xtm1.copy()
    # flow CoM x,y
    for i in range(2):
        loc[..., i] += dt * loc[..., 2 + i]
    # flow marker x,y
    N = len(loc.T)
    M = (N - 5) // 2
    r = sqrt(loc[..., 5::2]**2 + loc[..., 6::2]**2)
    thview = loc[..., 5:].reshape(loc.shape[:-1] + (M, 2))
    th = arctan2(thview.T[0], thview.T[1]).T
    thdot = loc[..., [4]]
    loc[..., 5::2] += r * (cos(dt * thdot + th) - cos(th))
    loc[..., 6::2] += r * (sin(dt * thdot + th) - sin(th))
    # flow ydot
    loc[..., 3] -= dt

    out[...] = normal(loc=loc, scale=scale)
    return out


def pxt(xtm1, u, dt, scale):
    """direct-return state transition probability"""
    out = zeros_like(xtm1)
    return pxt_rbr(xtm1, u, out, dt, scale)


def pzt_rbr(zt, xt, out):
    """"Probability" that observations zt came from state xt. Implemented as a
    cost function of RBT prediction error of zt.
    WARN: for relative marker positions
    INPUTS:
        zt -- Mx1 -- observation of M observables quantities
        xt -- PxN -- P estimates of N states

    NOTES:
        Since zt is a rigid point on body xt, augment xt with RBT to zt.
        The particle filter renormalizes the probability of the particles,
        so the output of this function doesn't need to cleanly integrate to
        1. This lets us return 1/(1+err), where `err` is the Euclidean
        distance between the observed marker and its predicted location by the
        RBTs. The form of the function makes low errors preferred while
        avoiding division by 0/numeric instability.
    """
    if not isscalar(out):
        out[...] = 0
    n = len(xt.T) - 5
    d = zt[...] - xt[..., [0, 1] * (n // 2)] - xt[..., 5:n + 5]
    out += sum(d**2, -1)
    # pairwise distance error
    pairs = list(combinations(range(0, len(xt.T) - 5, 2), 2))
    for i1, i2 in pairs:
        k1, k2 = i1 + 5, i2 + 5
        dz = (zt[..., [i1, i1 + 1]] - zt[..., [i2, i2 + 1]])**2
        dx = (xt[..., [k1, k1 + 1]] - xt[..., [k2, k2 + 1]])**2
        out += sum((dz - dx)**2, -1)

    out *= 1000
    out += 1
    if not isscalar(out):
        out[...] = 1 / out
    else:
        out = 1 / out
    return out


def pzt(zt, xt):
    """direct-return observation probability"""
    out = zeros(len(xt))
    if xt.ndim == 1:
        out = 0
    return pzt_rbr(zt, xt, out)


def construct_pf(M, N, dt, scale):
    """Construct Particle Filter for falling disk given state space and
    observation space size.
    INPUTS:
        M -- observation space size
        N -- state space size
        dt -- time step of observations
        scale -- N tuple of variances for pxt forward flowing
    """
    pardict = {'pxt_pars': [dt, scale], 'vec': True}
    return ParticleFilterFactory(pxt, pzt, **pardict)

# ===================================================================
# Unit Test


if __name__ == '__main__':
    # raise NotImplementedError('TODO')
    # Simulation
    dt = 0.01
    t1 = 10
    npts = 3
    q0 = (0, 0, 0)   # x,y,theta
    xi0 = (0, 0, 5)  # xdot,ydot,thetadot
    t = arange(0, t1, dt)
    gcom, out, obs = compute_motion(r_[q0, xi0], t, npts)

    # Matrices
    L, M = len(t), 2 * npts
    N = M + 5
    P = 10
    scale = [0.1, 0.1, 0.1, 0.1, 1] + [0.01] * M

    rbr = 0
    vec = 0
    pf = ParticleFilterFactory(pxt, pzt, pxt_pars=[dt, scale], vec=vec)
    if rbr:
        pf = ParticleFilterFactory(pxt_rbr, pzt_rbr, pxt_pars=[
                                   dt, scale], vec=vec, P=P, N=N, rbr=rbr)
    X_t = zeros((L, P, N))
    # X_t[-1] = [0, 0, 2, 0, 0] + list(obs.T[0].flatten())
    print('Starting particle filter...')
    tref = time()
    seed(0)
    for i, _ in enumerate(t):
        X_t[i] = pf(X_t[i - 1], 0, obs.T[i].flatten())
    print(f'Done! t={time()-tref:.2f}s')

    # est = mu_t
    # tru = reconstruct(est, out, obs)
    # plots(t, tru, est)
