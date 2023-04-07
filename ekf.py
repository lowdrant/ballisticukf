#!/usr/bin/env python3
"""
Run simulation and EKF estimation.

Choose units (mass, length, time) s.t. m = r = g = I = 1
"""
from time import time

from numba import njit, prange
from numpy import (arange, arctan2, cos, eye, fill_diagonal, r_, sin, sqrt,
                   zeros)
from numpy.random import seed

from filters import *
from helpers import *

# Simulation
dt = 0.01
t1 = 5
npts = 3
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
# TODO: remove unnecessary vectorization


def g_rbr(u, mu, mubar):
    """state transition function
    INPUTS:
        u -- input vector
        xtm1 -- Mx1 -- M states at time t-1.
                       Format: (x,y,vx,vy,w,dmx0,dmy0,...,dmxN,dmyN)
        xt -- Nx1 -- M states at time t. Format as xtm1

    NOTES:
        Model: disk falling due to gravity with makers at some distance and
               angle from CoM.
    """
    mubar[...] = mu.copy()
    # CoM Motion
    mubar[..., 0] += mu[..., 2] * dt
    mubar[..., 1] += mu[..., 3] * dt
    mubar[..., 3] -= dt
    # Marker Motion
    N = len(mubar.T)
    M = (N - 5) // 2
    # -- intermediate memory allocation --
    mx = mu[..., 5::2]
    my = mu[..., 6::2]
    thdot = mubar[..., 4]
    s = sin(dt * thdot)
    c = cos(dt * thdot)
    # -- intermediate memory allocation --
    mubar[..., 5::2] = mx * c - my * s
    mubar[..., 6::2] = mx * s + my * c
    return mubar


def g(u, mu):
    mubar = mu.copy()
    return g_rbr(u, mu, mubar)


def G_rbr(u, mu, G_t):
    """State transition jacobian
    INPUTS:
        u -- input vector
        mu -- Nx1 -- estimated state mean
        G_t -- NxN -- return by ref jacobian
    """
    # Const values (self-dependence, velocities)
    G_t[...] = 0
    fill_diagonal(G_t, 1)  # mem-efficient identity matrix insert
    G_t[0, 2] = dt  # x depends on vx
    G_t[1, 3] = dt  # y depends on vy

    # -- intermediate memory allocation --
    mx = mu[..., 5::2]
    my = mu[..., 6::2]
    thdot = mu[..., 4]
    s = sin(dt * thdot)
    c = cos(dt * thdot)
    # -- intermediate memory allocation --
    # TODO: take advantage of precomputes

    # Marker dx
    for i in range(5, len(G_t), 2):
        G_t[i, i] = c  # dx on dx
        G_t[i, i + 1] = - s  # dx on dy
    G_t[5::2, 4] = -mx * s - my * c
    G_t[5::2, 4] *= dt
    # Marker dy
    for i in range(6, len(G_t), 2):
        G_t[i, i - 1] = s  # dy on dx
        G_t[i, i] = c  # dy on dy
    G_t[6::2, 4] = mx * c - my * s
    G_t[6::2, 4] *= dt

    return G_t


def G(u, mu):
    N = len(mu.T)
    G_t = zeros((N, N))
    return G_rbr(u, mu, G_t)

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
    out[...] = mubar_t[..., 5:]
    out[..., 5::2] += mubar_t[..., 0]
    out[..., 6::2] += mubar_t[..., 1]
    return out


def h(mubar_t):
    out = mubar_t[..., 5:].copy()
    return h_rbr(mubar_t, out)


# Observation Jacobian
# - obs_x[i] = x + mx[i]
# - obs_y[i] = y + my[i]
H = zeros((M, N))
H[::2, 0] = 1
H[1::2, 0] = 1
H[:, 5:] = eye(M)

# ===================================================================
# Estimation

rbr = 0
do_njit = 1
callrbr = 0

kwargs = {'rbr': rbr, 'njit': do_njit, 'callrbr': callrbr}
ekf = EKF(g, h, G, H, R, Q, **kwargs)
if rbr:
    ekf = EKF(g_rbr, h_rbr, G_rbr, H, R, Q, **kwargs)

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
