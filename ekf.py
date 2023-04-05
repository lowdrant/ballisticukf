#!/usr/bin/env python3
"""
Run simulation and EKF estimation.

Choose units (mass, length, time) s.t. m = r = g = I = 1
"""
from time import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import normal, rand, seed

from filters import *
from helpers import *

# Constants
dt = 0.01
t1 = 100
npts = 5
q0 = (0, 0, 0)   # x,y,theta
xi0 = (0, 0, 0.5)  # xdot,ydot,thetadot
t = arange(0, t1, dt)
gcom, out, obs = compute_motion(r_[q0, xi0], t, npts)

N, K = len(t), 2 * npts
M = K + 5

scale = [0.01] * 2 + [0.0001] * 2 + [0.1] + [0.000001] * (2 * npts)

# Model Uncertainty Covariance
R = eye(M)
# Measurement Noise Covariance
Q = eye(K)

# ===================================================================
# State Transitions


def g(xtm1):
    """state transition function
    INPUTS:
        xtm1 -- NxM -- N estimates of M states at time t-1.
                       Format: (x,y,vx,vy,w,mx0,my0,...,mxN,myN)
    OUTPUTS:
        xt -- NxM -- N estimates of M states at time t. Format as xtm1

    NOTES:
        Model: disk falling due to gravity with makers at some distance and
               angle from CoM.
    """
    xt = array(xtm1, copy=1)
    # CoM Motion
    xt[..., 0] += xtm1[..., 2] * dt
    xt[..., 1] += xtm1[..., 3] * dt
    xt[..., 3] -= dt
    # Marker Motion
    M = len(xt.T)
    numpts = (M - 5) // 2
    r = sqrt(xt[..., 5:M:2]**2 + xt[..., 6:M:2]**2)
    thview = xt[..., 5:].reshape(xt.shape[:-1] + (numpts, 2))
    th = arctan2(*thview.T).T
    xt[..., 5:M:2] += r * (cos(dt * xt[..., 4] + th) - cos(th))
    xt[..., 6:M:2] += r * (sin(dt * xt[..., 4] + th) - sin(th))

    return xt


def h(mubar_t):
    """state observation function
    INPUTS:
        mubar_t -- ...NxM -- N estimates of M states at time t.
                             Format: (x,y,vx,vy,w,mx0,my0,...,mxN,myN)
    OUTPUTS:
        hat_zt -- ...Nx(M-5) -- N estimates of M-5 observations at time t.
                                Format: (mx0,my0,...,mxN,myN)
    """
    mubar_t = asarray(mubar_t)
    return mubar_t[..., 5:]


def G(xtm1, out=None):
    """Jacobian of g.
    INPUTS:
        xtm1 -- ...NxM -- N estimates of M states at time t-1.
                          Format: (x,y,vx,vy,w,mx0,my0,...,mxN,myN)
    OUTPUTS:
       Gt -- ...MxM -- jacoban of g at xtm1 

    NOTES:
        Lest I forget:
            https://www.wolframalpha.com/input
            ?i=partial+derivative+sqrt
            %28x%5E2%2By%5E2%29+*+y+%2F+%28sqrt%28x%5E2%2By%5E2%29%29
    """
    xtm1 = asarray(xtm1)
    n_mx, n_my = range(5, len(xtm1.T), 2), range(6, len(xtm1.T), 2)
    if out is None:
        out = zeros(xtm1.shape + (len(xtm1.T),), dtype=float)
    out[...] = 0.  # force to 0 float
    out[..., 0, 2] = dt  # x on vx
    out[..., 1, 3] = dt  # y on vy
    out[..., n_mx, n_my] = 1  # my marker dependence
    out[..., n_my, n_mx] = 1  # mx marker dependence
    out[..., 5:, 4] = 1  # theta_dot marker dependence
    return out


def H(mubar_t, out=None):
    """Jacobian of h.
    INPUTS:
        mubar_t -- NxM -- N estimates of M states at time t-1.
                          Format: (x,y,vx,vy,w,mx0,my0,...,mxN,myN)
    OUTPUTS:
        Ht -- (M-5)xM -- jacoban of h at mubar_t
    """
    mubar_t = asarray(mubar_t)
    if out is None:
        out = zeros(mubar_t.shape[:-2] + (M - 5, M), dtype=float)
    out[...] = 0.
    out[..., 5:] = eye(M - 5)
    return out


# ===================================================================
# Estimation

# Filtering
print('Starting EKF...')
tref = time()
ekf = ExtendedKalmanFilter(g, h, G, H, R, Q)
mu_t = zeros((N, M))
mu_t[-1] = [1, 0, 0, 0, 0] + list(obs.T[0].flatten())
sigma_t = zeros((len(t), M, M))  # TODO: initialize better
seed(0)
for i, _ in enumerate(t):
    mu_t[i], sigma_t[i] = ekf(mu_t[i - 1], sigma_t[i - 1], obs.T[i].flatten())
print(f'Done! t={time()-tref:.2f}s')

# ===================================================================
# Validation

est = mu_t
tru = reconstruct(est, out, obs)
plots(t, tru, est)

