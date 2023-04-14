#!/usr/bin/env python3
"""
EKF construction for main.py
"""
__all__ = ['construct_ekf']
from numpy import cos, eye, fill_diagonal, sin, zeros

from .EstimatorFactory import EKFFactory
from .helpers import *

# ===================================================================
# State Transitions


def g_rbr(t, u, mu, dt, mubar):
    """state transition function
    INPUTS:
        u -- input vector
        xtm1 -- Nx1 -- M states at time t-1.
                       Format: (x,y,vx,vy,w,dmx0,dmy0,...,dmxN,dmyN)
        xt -- Nx1 -- M states at time t. Format as xtm1

    NOTES:
        Model: disk falling due to gravity with makers at some distance and
               angle from CoM.
    """
    mubar[...] = 0
    # CoM Motion
    mubar[0] = mu[0] + mu[2] * dt
    mubar[1] = mu[1] + mu[3] * dt
    mubar[3] = mu[3] - dt
    mubar[4] = mu[4]
    # Marker Motion
    # - extract views
    mx, my, thdot = mu[5::2], mu[6::2], mu[4]
    # -- intermediate memory allocation --
    s = sin(dt * thdot)
    c = cos(dt * thdot)
    # -- intermediate memory allocation --
    mubar[5::2] = mx * c - my * s
    mubar[6::2] = mx * s + my * c
    return mubar


def g(t, u, mu, dt):
    """direct-return state transition function"""
    mubar = mu.copy()
    return g_rbr(t, u, mu, dt, mubar)


def G_rbr(t, u, mu, dt, G_t):
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

    # extract consts as views
    mx = mu[5::2]
    my = mu[6::2]
    thdot = mu[4]
    # -- intermediate memory allocation --
    s = sin(dt * thdot)
    c = cos(dt * thdot)
    # -- intermediate memory allocation --

    # Marker dx
    fill_diagonal(G_t[5::2, 5::2], c)  # dx on dx
    fill_diagonal(G_t[5::2, 6::2], -s)  # dx on dy
    G_t[5::2, 4] = dt * (-mx * s - my * c)  # dx on thdot
    # Marker dy
    fill_diagonal(G_t[6::2, 5::2], s)  # dy on dx
    fill_diagonal(G_t[6::2, 6::2], c)  # dy on dy
    G_t[6::2, 4] = dt * (mx * c - my * s)  # dy on thdot

    return G_t


def G(t, u, mu, dt):
    """direct-return state transition jacobian"""
    N = len(mu.T)
    G_t = zeros((N, N))
    return G_rbr(t, u, mu, dt, G_t)


def h_rbr(t, mubar_t, zhat):
    """state observation function
    INPUTS:
        mubar_t -- Mx1 -- M states at time t.
                          Format: (x,y,vx,vy,w,dmx0,dmy0,...,dmxN,dmyN)
    OUTPUTS:
        zhat -- (M-5)x1 -- M-5 observations at time t.
                           Format: (dmx0,dmy0,...,dmxN,dmyN)
    """
    zhat[...] = 0
    zhat[::2] = mubar_t[5::2] + mubar_t[0]
    zhat[1::2] = mubar_t[6::2] + mubar_t[1]
    return zhat


def h(t, mubar_t):
    """direct-return observation function"""
    out = mubar_t[..., 5:].copy()
    return h_rbr(t, mubar_t, out)


def construct_ekf(k, n, dt, njit, Q=None, R=None):
    """Construct EKF for falling disk given state space and observation
    space size.
    INPUTS:
        k -- observation space size
        n -- state space size
        njit -- bool, optional -- enable njit optimization, default: False
        Q -- MxM, optional -- specify Q, see code for default
        R -- NxN, optional -- specify R, see code for default
    """
    if R is None:
        R = eye(n) * 0.1
        R[4, 4] = 1  # unknown thetadot
        R[0, 2] = 1  # x depends on vx
        R[2, 0] = 1
        R[1, 3] = 1  # y depends on vy
        R[3, 1] = 1
    if Q is None:
        Q = eye(k)
        Q[...] = 0
    H = zeros((k, n))
    H[::2, 0] = 1
    H[1::2, 1] = 1
    H[:, 5:] = eye(k)  # H defined here since 'M' is determined by main.py

    pardict = {}
    for key in ('g', 'G'):
        pardict[key + '_pars'] = [dt]
    return EKFFactory(g, h, G, H, R, Q, n=n, k=k, njit=njit, **pardict)
