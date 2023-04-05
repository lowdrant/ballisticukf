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

N, M = len(t), 5 + 2 * npts

scale = [0.01] * 2 + [0.0001] * 2 + [0.1] + [0.000001] * (2 * npts)

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
    n = len(xt.T)
    # CoM Motion
    xt[..., 0] += xtm1[..., 2] * dt
    xt[..., 1] += xtm1[..., 3] * dt
    xt[..., 3] -= dt
    # Marker Motion
    r = sqrt(xt[..., 5:n:2]**2 + xt[..., 6:n:2]**2)
    th = arctan2(*xt.reshape(..., (n - 5) // 2, 2).T)
    xt[..., 5:n:2] += r * (cos(dt * xt[..., 4] + th) - cos(th))
    xt[..., 6:n:2] += r * (sin(dt * xt[..., 4] + th) - sin(th))

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
    r, c = range(5, len(xtm1.T), 2), range(6, len(xtm1.T), 2)
    if out is None:
        out = zeros(xtm1.shape + (len(xtm1.T),), dtype=float)
    out[...] = 0.  # force to 0 float
    out[..., 0, 2] = dt  # x on vx
    out[..., 1, 3] = dt  # y on vy
    out[..., r, c] = 1  # my marker dependence
    out[..., r, 4] = 1  # w marker dependence
    return out


def H(mubar_t, out=None):
    """Jacobian of h.
    INPUTS:
        mubar_t -- ...NxM -- N estimates of M states at time t-1.
                          Format: (x,y,vx,vy,w,mx0,my0,...,mxN,myN)
    OUTPUTS:
        Ht -- ...(M-5)xM -- jacoban of h at mubar_t
    """
    mubar_t = asarray(mubar_t)
    if out is None:
        out = zeros(mubar_t.shape + (len(mubar_t.T),), dtype=float)
    out[...] = 0.
    n = len(mubar_t.T)
    r, c = range(5, n, 2), range(6, n, 2)
    out[..., r, c] = 1.
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

# Reconstruction
est = mu_t.copy()
est[:, 5:] += est[:, [0, 1] * ((len(est.T) - 5) // 2)]
tru = zeros_like(est)
tru[:, :5] = out[:, [0, 1, 3, 4, 5]]  # 2 is theta; skip
tru[:, 5:] = obs.T.reshape(len(obs.T), prod([v for v in obs.T.shape[1:]]))

# ===================================================================
# Plot

kwest = {'ms': 2, 'lw': 0.5, 'alpha': 1}
axp = newplot('parametric motion')
axp.grid(0)
axp.plot(gcom[0, -1], gcom[1, -1], label='CoM', c='tab:blue')
axp.plot(*est.T[:2], '.-', label='estimated CoM', c='tab:blue', **kwest)
c = list(mcolors.TABLEAU_COLORS.keys())[2:]
for i in range(0, obs.shape[1] + 2, 2):
    k = i // 2
    axp.plot(*obs[:, k], c=c[k])  # ,label=f'mkr{k}')
    axp.plot(*est[:, [5 + i, 5 + i + 1]].T, '.', c=c[k], **kwest)
axp.legend(loc='upper left')
axp.set_xlabel('$x$')
axp.set_ylabel('$y$')
axp.set_aspect('equal')

lbls = ['$x$', '$y$', '$\\dot{x}$', '$\\dot{y}$', '$\\dot{\\theta}$']

num = 'state estimates'
plt.figure(num).clf()
_, axs = plt.subplots(nrows=5, sharex='all', num=num)
for i, ax in enumerate(axs):
    ax.plot(t, tru[:, i], '.-')
    ax.plot(t, est[:, i], '.-')
    lbl = lbls[i] if i < 5 else f'm{chr(ord("x") + (i - 5) % 2)}{(i - 5) // 2}'
    ax.set_ylabel(lbl)  # , rotation=0)
for a in axs:
    a.grid(1)
axs[-1].set_xlabel('$t$')
axs[0].set_title(axs[0].get_figure().get_label())

num = 'marker estimates'
plt.figure(num).clf()
_, axm = plt.subplots(nrows=obs.shape[0] * obs.shape[1], sharex='all', num=num)
for i, ax in enumerate(axm):
    ax.plot(t, tru[:, 5 + i], '.-')
    ax.plot(t, est[:, 5 + i], '.-')
    lbl = f'm{chr(ord("x") + i % 2)}{i // 2}'
    ax.set_ylabel(lbl)  # , rotation=0)
for a in axm:
    a.grid(1)
axm[-1].set_xlabel('$t$')
axm[0].set_title(axm[0].get_figure().get_label())

ax = newplot('rb params')
c = list(mcolors.TABLEAU_COLORS.keys())
for i in range(obs.shape[1]):
    k = 5 + 2 * i
    rtru = sqrt(sum((tru[..., [0, 1]] - tru[..., [k, k + 1]])**2, -1))
    rout = sqrt(sum((out[..., [0, 1]] - obs[..., i, :].T)**2, -1))
    rest = sqrt(sum((est[..., [0, 1]] - est[..., [k, k + 1]])**2, -1))
    ax.plot(t, rout, '--', c=c[i], lw=3)  # ,label=f'out {i})
    ax.plot(t, rtru, '.-', c=c[i], ms=2)  # ,label=f'true {i}')
    ax.plot(t, rest, 'x-', label=f'est {i}', c=c[i])
ax.legend(loc='upper left')

ipychk()
