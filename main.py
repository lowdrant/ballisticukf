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
t1 = 1
npts = 5
q0 = (0.1, -0.1, 0)   # x,y,theta
xi0 = (0, -0, 5)  # xdot,ydot,thetadot
M = 10000

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


def pxt(xtm1):
    """generate probability cloud of current state given prev state
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
    scale = [0.01] * 2 + [1] * 2 + [2] + [0.001] * (len(loc.T) - 5)
    # flow CoM x,y
    for i in range(2):
        loc[..., i] += dt * loc[..., 3 + i]
    # flow marker x,y
    for i in range(5, len(xtm1.T), 2):
        d = loc[..., [i, i + 1]] - loc[..., [0, 1]]
        theta = arctan2(*d.T)
        r = sqrt(sum(d**2, -1))
        loc[..., i] += dt * loc[..., 2] + dt * r * sin(theta) * loc[..., 4]
        loc[..., i + 1] += dt * loc[..., 3] + dt * r * cos(theta) * loc[..., 4]
    # flow ydot
    loc[..., 3] -= dt
    return normal(loc=loc, scale=scale)


def pzt(zt, xt):
    """"Probability" that observations zt came from state xt. Implemented as a
    cost function of RBT prediction error of zt.

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
    err = sum((zt - xt[..., 5:])**2, 1)
    return 1 / (1 + 100 * err)


# ===================================================================
# Filtering

obs = einsum('ij...,j->i...', gp, [0, 0, 1])[:-1]
ctrd = obs[:-1].mean(1).T

print('Starting particle filter...')
tref = time()
pf = ParticleFilter(pxt, pzt, dbg=1)
Xt = zeros((len(t), M, 5 + len(obs.T[0].flatten())))
Xt[-1] = [0, 0, 0, 0, 0] + list(obs.T[0].flatten())
seed(0)
for i, _ in enumerate(t):
    Xt[i] = pf(Xt[i - 1], obs.T[i].flatten())
print(f'Done! t={time()-tref:.2f}s')

# ===================================================================
# Reconstruction

est = mean(Xt, 1)
tru = zeros_like(est)
tru[:, :5] = out[:, [0, 1, 3, 4, 5]]
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
    print(i, k)
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
_, axm = plt.subplots(nrows=obs.shape[1], sharex='all', num=num)
for i, ax in enumerate(axm):
    ax.plot(t, tru[:, i], '.-')
    ax.plot(t, est[:, i], '.-')
    lbl = f'm{chr(ord("x") + i % 2)}{i // 2}'
    ax.set_ylabel(lbl)  # , rotation=0)
for a in axm:
    a.grid(1)
axm[-1].set_xlabel('$t$')
axm[0].set_title(axm[0].get_figure().get_label())

ipychk()

