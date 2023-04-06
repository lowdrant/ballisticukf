#!/usr/bin/env python3
"""
Run simulation and EKF estimation.

Choose units (mass, length, time) s.t. m = r = g = I = 1
"""
from time import time

from numba import njit
from numpy import arange, arctan2, cos, eye, r_, sin, sqrt, zeros
from numpy.random import seed

from filters import *
from helpers import *

# Simulation
dt = 0.01
t1 = 10
npts = 2
q0 = (0, 0, 0)   # x,y,theta
xi0 = (0, 0, 5)  # xdot,ydot,thetadot
t = arange(0, t1, dt)
gcom, out, obs = compute_motion(r_[q0, xi0], t, npts)

# Matrices
L, M = len(t), 2 * npts
N = M + 5
# model uncertainty
R = eye(N)
R[5::2, 4] = 1  # marker delta depends on theta_dot
R[0, 2] = 1  # x depends on vx
R[1, 3] = 1  # y depends on vy
# measurement uncertainty
Q = eye(M)

# ===================================================================
# State Transitions


def g_rbr(u, mu, mubar):
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
    mubar[...] = mu.copy()  # view(mu.dtype)
    # CoM Motion
    mubar[..., 0] += mu[..., 2] * dt
    mubar[..., 1] += mu[..., 3] * dt
    mubar[..., 3] -= dt
    # Marker Motion
    N = len(mubar.T)
    M = (N - 5) // 2
    r = sqrt(mubar[..., 5::2]**2 + mubar[..., 6::2]**2)
    thview = mubar[..., 5:].reshape(mubar.shape[:-1] + (M, 2))
    th = arctan2(thview.T[0], thview.T[1]).T
    mubar[..., 5::2] += r * (cos(dt * mubar[..., 4] + th) - cos(th))
    mubar[..., 6::2] += r * (sin(dt * mubar[..., 4] + th) - sin(th))
    return mubar


def g(u, mu):
    mubar = mu.copy()
    return g_rbr(u, mu, mubar)


# @njit
def h_rbr(mubar_t, out):
    """state observation function
    INPUTS:
        mubar_t -- ...NxM -- N estimates of M states at time t.
                             Format: (x,y,vx,vy,w,mx0,my0,...,mxN,myN)
    OUTPUTS:
        hat_zt -- ...Nx(M-5) -- N estimates of M-5 observations at time t.
                                Format: (mx0,my0,...,mxN,myN)
    """
    out[...] = mubar_t[..., 5:].copy()
    out[..., 5::2] += mubar_t[..., 0]
    out[..., 6::2] += mubar_t[..., 1]
    return out


def h(mubar_t):
    out = mubar_t[..., 5:].copy()
    return h_rbr(mubar_t, out)


# Jacobians
n_mx, n_my = range(5, M, 2), range(6, M, 2)
G = zeros((N, N))
G[0, 2] = dt
G[1, 2] = dt
G[n_mx, n_my] = 1
G[n_my, n_mx] = 1
G[5::2, 4] = -1
G[6::2, 4] = 1

H = zeros((M, N))
H[:, N - M:] = eye(M)

# ===================================================================
# Estimation

rbr = 1
do_njit = 0
callrbr = 0

kwargs = {'rbr': rbr, 'njit': do_njit, 'callrbr': callrbr}
ekf = EKF(g, h, G, H, R, Q, **kwargs)
if rbr:
    ekf = EKF(g_rbr, h_rbr, G, H, R, Q, **kwargs)

# Filtering
mu_t, sigma_t = zeros((L, N)), zeros((L, N, N))
mu_t[-1] = [1, 0, 0, 0, 0] + list(obs.T[0].flatten())
print('Starting EKF...')
tref = time()
seed(0)
if callrbr:
    for i, _ in enumerate(t):
        ekf(mu_t[i - 1], sigma_t[i - 1], 0,
            obs.T[i].flatten(), mu_t[i], sigma_t[i])
else:
    for i, _ in enumerate(t):
        mu_t[i], sigma_t[i] = ekf(
            mu_t[i - 1], sigma_t[i - 1], 0, obs.T[i].flatten())
print(f'Done! t={time()-tref:.2f}s')

# ===================================================================
# Validation

est = mu_t
tru = reconstruct(est, out, obs)
plots(t, tru, est)
