from itertools import combinations

from numpy import *
from numpy.random import normal

from filters import ParticleFilter
from helpers import *

# ===================================================================
# Priors


def pxt(xtm1, dt, scale):
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
    return normal(loc=loc, scale=scale)


def pzt(zt, xt):
    """"Probability" that observations zt came from state xt. Implemented as a
    cost function of RBT prediction error of zt.
    WARN: for relative marker positions
    INPUTS:
        zt -- PxM -- P observations of M observables quantities
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
    xt, zt = asarray(xt), asarray(zt)
    xt = xt[newaxis, :] if xt.ndim == 1 else xt
    zt = zt[newaxis, :] if zt.ndim == 1 else zt
    err = zeros(len(xt))
    # coordinate error
    n = len(xt.T) - 5
    d = zt[...] - xt[..., [0, 1] * (n // 2)] - xt[..., 5:n + 5]
    err += sum(d**2, -1)
    # pairwise distance error
    pairs = list(combinations(range(0, len(xt.T) - 5, 2), 2))
    for i1, i2 in pairs:
        k1, k2 = i1 + 5, i2 + 5
        dz = (zt[..., [i1, i1 + 1]] - zt[..., [i2, i2 + 1]])**2
        dx = (xt[..., [k1, k1 + 1]] - xt[..., [k2, k2 + 1]])**2
        err += sum((dz - dx)**2, -1)
    return 1 / (1 + 1000 * err)


def construct_pf(M, N, dt, scale):
    """Construct Particle Filter for falling disk given state space and
    observation space size.
    INPUTS:
        M -- observation space size
        N -- state space size
        dt -- time step of observations
        scale -- N tuple of variances for pxt forward flowing
    """
    pardict = {'pxt_args': [dt, scale]}
    return ParticleFilter(pxt, pzt, **pardict)

# ===================================================================
# Unit Test


if __name__ == '__main__':

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

    ekf = construct_ekf(M, N, dt, 1)
    mu_t, sigma_t = zeros((L, N)), zeros((L, N, N))
    sigma_t[-1] = 10
    fill_diagonal(sigma_t[-1, :5, :5], 10)
    sigma_t[-1, 4, 4] = 100
    mu_t[-1] = [0, 0, 2, 0, 0] + list(obs.T[0].flatten())
    print('Starting EKF...')
    tref = time()
    seed(0)
    for i, _ in enumerate(t):
        mu_t[i], sigma_t[i] = ekf(
            mu_t[i - 1], sigma_t[i - 1], 0, obs.T[i].flatten())
    print(f'Done! t={time()-tref:.2f}s')

    est = mu_t
    tru = reconstruct(est, out, obs)
    plots(t, tru, est)
