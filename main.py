#!/usr/bin/env python3
"""
Run simulation and estimation. Makes use of try/excepts to skip already
completed computations in ipython debugging sessions

Choose units (mass, length, time) s.t. m = r = g = I = 1
"""
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import normal, rand, seed
from time import time
from filters import *
from helpers import *
import matplotlib.colors as mcolors

# Constants
dt = 0.01
t1 = 100
npts = 5
q0 = (0, 0, 0)   # x,y,theta
xi0 = (0, 0, 0.5)  # xdot,ydot,thetadot
M = 100

use_rel = 1
scale = [0.01] * 2 + [0.0001] * 2 + [0.1] + [0.000001] * (2 * npts)

# ===================================================================
# Simulate Point Motion

# Point Observations
seed(0)
tf = []
for i in range(npts):
    r = rand()
    th = 2 * pi * rand()
    tf.append(gen_transform(r, th))
tf = asarray(tf)

# CoM Motion
x0 = r_[q0, xi0]
t = arange(0, t1, dt)
gcom, out = sim(x0, t, return_state=1)

# Point Motion
gp = einsum('ijk,njm->imnk', gcom, tf)

# Simple Validation
for i in range(npts):
    d2 = (gcom[0, -1] - gp[0, -1, i])**2 + (gcom[1, -1] - gp[1, -1, i])**2
    assert sum(abs(diff(d2))) < 1e-6, 'rigid body assumption violated!'

# ===================================================================
# Priors


def pxt_rel(xtm1):
    """generate probability cloud of current state given prev state
    WARN: for relative marker positions
    INPUTS:
        xtm1 -- NxM -- N prior estimates for M states
    NOTES:
        Modeling motion of off-axis point, since that's what we are observing

        For a disk at (x,y) moving at (vx,vy) with angle a and angular
        velocity w, a point at radius r will have
        observed velocity vo (vx+w*r*sina, vy+w*r*cosa)
    """
    loc = array(xtm1, copy=1)
    loc = loc[newaxis, ...] if loc.ndim == 1 else loc
    # flow CoM x,y
    for i in range(2):
        loc[..., i] += dt * loc[..., 2 + i]
    # flow marker x,y
    for i in range(5, len(xtm1.T), 2):
        th = arctan2(*loc[..., [i, i + 1]].T).T
        r = sqrt(sum(loc[..., [i, i + 1]]**2, -1))
        loc[..., i] += r * (cos(dt * loc[..., 4] + th) - cos(th))
        loc[..., i + 1] += r * (sin(dt * loc[..., 4] + th) - sin(th))
    # flow ydot
    loc[..., 3] -= dt
    return normal(loc=loc, scale=scale)


def pxt_abs(xtm1):
    """generate probability cloud of current state given prev state
    WARN: for absolute marker positions
    SEE ALSO: pxt_rel
    INPUTS:
        xtm1 -- NxM -- N prior estimates for M states
    OUTPUTS:
        out -- NxM -- N current estimates for M states
    NOTES:
        - `pxt_rel` performs actual computations
        - 
    """
    loc = array(xtm1, copy=1)
    prev_v = loc[..., [2, 3]].copy()
    # markers to rel distance
    for i in range(5, len(loc.T)):
        loc[..., i] -= loc[..., (i + 1) % 2]
    out = pxt_rel(loc)
    # markers to abs distance
    for i in range(5, len(loc.T)):
        k = (i + 1) % 2
        out[..., i] += out[..., k] + dt * prev_v[..., k]
    return out


def pzt_abs(zt, xt):
    """"Probability" that observations zt came from state xt. Implemented as a
    cost function of RBT prediction error of zt.
    WARN: for absolue marker positions
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
    err = sum((zt - xt[..., 5:])**2, -1)
    # pairwise distance error
    # '''
    pairs = list(combinations(range(0, len(xt.T) - 5, 2), 2))
    for i1, i2 in pairs:
        k1, k2 = i1 + 5, i2 + 5
        dz = (zt[..., [i1, i1 + 1]] - zt[..., [i2, i2 + 1]])**2
        dx = (xt[..., [k1, k1 + 1]] - xt[..., [k2, k2 + 1]])**2
        err += sum((dz - dx)**2, -1)
    # '''
    return 1 / (1 + 100 * err)


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
    # '''
    n = len(xt.T) - 5
    d = zt[...] - xt[..., [0, 1] * (n // 2)] - xt[..., 5:n + 5]
    err += sum(d**2, -1)
    # '''
    # pairwise distance error
    pairs = list(combinations(range(0, len(xt.T) - 5, 2), 2))
    for i1, i2 in pairs:
        k1, k2 = i1 + 5, i2 + 5
        dz = (zt[..., [i1, i1 + 1]] - zt[..., [i2, i2 + 1]])**2
        dx = (xt[..., [k1, k1 + 1]] - xt[..., [k2, k2 + 1]])**2
        err += sum((dz - dx)**2, -1)
    # '''
    return 1 / (1 + 100 * err)


# ===================================================================
# Filtering

obs = einsum('ij...,j->i...', gp, [0, 0, 1])[:-1]
ctrd = obs.mean(1).T

print('Starting particle filter...')
tref = time()
pf = ParticleFilter(pxt_abs, pzt_abs)
if use_rel:
    pf = ParticleFilter(pxt_rel, pzt_rel)
Xt = zeros((len(t), M, 5 + len(obs.T[0].flatten())))
Xt[-1] = [1, 0, 0, 0, 0] + list(obs.T[0].flatten())
seed(0)
for i, _ in enumerate(t):
    Xt[i] = pf(Xt[i - 1], obs.T[i].flatten())
print(f'Done! t={time()-tref:.2f}s')

# ===================================================================
# Reconstruction

est = mean(Xt, 1)
if use_rel:
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

