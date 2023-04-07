#!/usr/bin/env python3
"""
Run simulation and estimation.

Choose units (mass, length, time) s.t. m = r = g = I = 1
"""
from itertools import combinations
from time import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numba import njit
from numpy import *
from numpy.random import normal, rand, seed

from filters import *
from helpers import *

# Constants
dt = 0.01
t1 = 100
npts = 5
q0 = (0, 0, 0)   # x,y,theta
xi0 = (0, 0, 4)  # xdot,ydot,thetadot
M = 1000

t = arange(0, t1, dt)
gcom, out, obs = compute_motion(r_[q0, xi0], t, npts)

scale = [0.0001] * 2 + [0.0001] * 2 + [5] + [0.1] * (2 * npts)

# ===================================================================
# Priors


def pxt_rel(xtm1):
    """generate probability cloud of current state given prev state
    WARN: for relative marker positions
    INPUTS:
        xtm1 -- NxM -- N prior estimates for M states
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


def pzt_rel(zt, xt):
    """"Probability" that observations zt came from state xt. Implemented as a
    cost function of RBT prediction error of zt.
    WARN: for relative marker positions
    INPUTS:
        zt -- NxK -- N observations of K observables quantities
        xt -- NxM -- N estimates of M states

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


# ===================================================================
# Filtering

print('Starting particle filter...')
tref = time()
pf = ParticleFilter(pxt_rel, pzt_rel)
Xt = zeros((len(t), M, 5 + len(obs.T[0].flatten())))
Xt[-1] = [1, 0, 0, 0, 0] + list(obs.T[0].flatten())
seed(0)
for i, _ in enumerate(t):
    Xt[i] = pf(Xt[i - 1], obs.T[i].flatten())
print(f'Done! t={time()-tref:.2f}s')

# ===================================================================
# Validation

est = mean(Xt, 1)
tru = reconstruct(est, out, obs)
# plots(t, tru, est)
