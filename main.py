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

# Constants
dt = 0.01
t1 = 1
npts = 5
q0 = (0.1, -0.1, 0)   # x,y,theta
xi0 = (0.1, -0.1, 7)  # xdot,ydot,thetadot
M = 100

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
    if loc.ndim == 1:
        loc = loc[newaxis, ...]
    for i in range(2):
        loc[..., i] += dt * loc[..., 3 + i]
    theta = arctan2(loc[..., -1] - loc[..., 1], loc[..., -2] - loc[..., 0])
    r = sqrt(sum(loc[..., -2:]**2, 1))
    loc[..., -2] += dt * loc[..., 2] + dt * r * sin(theta) * loc[..., -3]
    loc[..., -1] += dt * loc[..., 3] + dt * r * cos(theta) * loc[..., -3]
    loc[..., 3] -= dt
    scale = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
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
    if xt.ndim == 1:
        xt = xt[newaxis, :]
    if zt.ndim == 1:
        zt = zt[newaxis, :]
    err = sum((zt - xt[..., -2:])**2, 1)
    return 1 / (1 + 100 * err)


# ===================================================================
# Filtering

obs = einsum('ij...,j->i...', gp, [0, 0, 1])
ctrd = obs[:-1].mean(1).T

print('Starting particle filter...')
tref = time()
pf = ParticleFilter(pxt, pzt, dbg=1)
Xt = zeros((len(t), M, 7))
Xt[-1] = [*ctrd[0], 0, 0, 0, 0, 0]
seed(0)
for i, _ in enumerate(t):
    Xt[i] = pf(Xt[i - 1], ctrd[i])
print(f'Done! t={time()-tref:.2f}s')

# ===================================================================
# Reconstruction

est = mean(Xt, 1)
tru = zeros_like(est)
tru[:, :5] = out[:, [0, 1, 3, 4, 5]]
tru[:, [5, 6]] = out[:, [0, 1]] - ctrd

est_ctrd = est[:, [-2, -1]]

# ===================================================================
# Plot

lbls = ['$x$', '$y$', '$\\dot{x}$', '$\\dot{y}$', '$\\dot{\\theta}$',
        '$dx$', '$dy$']

axp = newplot('parametric motion')
axp.grid(0)
axp.plot(gcom[0, -1], gcom[1, -1], label='CoM')
axp.plot(*ctrd.T, label='marker')
axp.plot(*est.T[:2], '.-', label='estimated CoM')
axp.plot(*est_ctrd.T, '.-', label='estimated marker')
axp.legend(loc='lower left')
axp.set_xlabel('$x$')
axp.set_ylabel('$y$')
axp.set_aspect('equal')

num = 'filter output'
plt.figure(num).clf()
_, axf = plt.subplots(nrows=len(Xt.T), sharex='all', num=num)
for i, ax in enumerate(axf):
    ax.plot(t, tru[:, i], '.-')
    ax.plot(t, est[:, i], '.-')
    ax.set_ylabel(lbls[i])
for a in axf:
    a.grid()
axf[-1].set_xlabel('$t$')
axf[0].set_title(a.get_figure().get_label())

num = 'pct err'
axpct = newplot(num)
axpct.grid(0)
axpct.plot(t, zeros_like(t), 'k--', lw=3)
for i in range(len(Xt.T) - 2):
    pe = 100 * (tru[:, i] - est[:, i]) / tru[:, i]
    axpct.plot(t, pe, '.-', label=lbls[i])
axpct.legend(loc='upper right')
axpct.set_xlabel('$t$')
axpct.set_ylabel('percent error')

ipychk()

