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
dt = 0.05
t1 = 3
npts = 5
q0 = (0.1, -0.1, 0)   # x,y,theta
xi0 = (0.1, 0, 5)  # xdot,ydot,thetadot
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
# Filtering
# TODO: vectorize probability functions

obs = einsum('ij...,j->i...', gp, [0, 0, 1])
ctrd = obs[:-1].mean(1)  # maybe centroid will be easier to work with?


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
    loc[..., 0] += dt * loc[..., 3] * \
        loc[..., 5] * loc[..., 6] * sin(loc[..., 2])
    loc[..., 1] += dt * loc[..., 4] * \
        loc[..., 5] * loc[..., 6] * cos(loc[..., 2])
    loc[..., 2] += dt * loc[..., 4]
    loc[..., 4] -= dt
    scale = [0.1, 0.1, pi / 2, 0.1, 0.1, 0.1, 0.1]
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
    r = xt[..., -1].T  # marker in body frame
    x, y, th = xt[..., :3].T  # body in world frame
    ctrd = c_[r, zeros_like(r), ones_like(r)].T
    zt_hat = einsum('ijk,jk->ik', SE2(x, y, th), ctrd)
    err = sum((zt.T - zt_hat[:-1])**2, 0)
    return 1 / (1 + 100 * err)


print('Starting particle filter...')
tref = time()
pf = ParticleFilter(pxt, pzt, dbg=1)
Xt = zeros((len(t), M, 7))
Xt[-1] = [*ctrd[..., 0], 0, 0, 0, 1, 0.5]
seed(0)
for i, _ in enumerate(t):
    Xt[i] = pf(Xt[i - 1], ctrd[..., i])
print(f'Done! t={time()-tref:.2}s')

# ===================================================================
# Reconstruction

estmean = mean(Xt, 1)
estmed = median(Xt, 1)

estgWB = zeros((3, 3, len(estmean)), dtype=float)
estgWB[:-1, :-1] = r2d(estmean[:, 2])
estgWB[0, -1] = estmean[:, 0]
estgWB[1, -1] = estmean[:, 1]
estgWB[-1, -1] = 1

estgBC = zeros_like(estgWB)
estgBC[[0, 1, 2], [0, 1, 2]] = 1
estgBC[0, -1] = estmean[:, -1]

est_ctrd = einsum('ijk,jmk->imk', estgWB, estgBC)[[0, 1], -1]

dtrue = sqrt(sum((out[:, [0, 1]] - ctrd.T)**2, 1))
dest = sqrt(sum((estmean[:, [0, 1]] - est_ctrd.T)**2, 1))

estr = estmean[:, -1]
mo = zeros_like(out)
mo[:, [0, 1]] = ctrd.T
mo[:, 3] = out[:, 3] + out[:, 5] * estr * sin(out[:, 2])
mo[:, 4] = out[:, 4] + out[:, 5] * estr * cos(out[:, 2])
mo[:, 5] = out[:, 5]

# ===================================================================
# Plot

axp = newplot('parametric motion')
axp.grid(0)
axp.plot(gcom[0, -1], gcom[1, -1], label='CoM')
axp.plot(*ctrd, label='marker centroid')
axp.plot(*estmean.T[:2], '.-', label='pf output')
# axp.plot(*est_ctrd, '.-', label='estimated marker')
axp.legend(loc='lower left')
axp.set_xlabel('$x$')
axp.set_ylabel('$y$')
axp.set_aspect('equal')

num = 'filter output'
plt.figure(num).clf()
ylbl = ['$x$', '$y$', '$\\theta$', '$\\dot{x}$', '$\\dot{y}$',
        '$\\dot{\\theta}$', '$r$']
_, axf = plt.subplots(nrows=Xt.shape[-1], sharex='all', num=num)
for i, ax in enumerate(axf[:-1]):
    ax.plot(t, mo[:, i], '.-', label='ground truth')
    ax.plot(t, estmean[:, i], '.-', label='estimate')
    ax.set_ylabel(ylbl[i])
axf[-1].plot(t, sqrt(sum((out[:, :2] - ctrd.T)**2, 1)), '.-')
axf[-1].plot(t, estmean[:, -1], '.-')
axf[-1].set_ylabel('dist')
for a in axf:
    a.grid()
axf[-1].set_xlabel('$t$')
axf[0].set_title(a.get_figure().get_label())


num = 'err'
axp = newplot(num)
se = sqrt(sum((ctrd - estmean.T[:2])**2, 0))
rmse = cumsum(se) / range(1, len(se) + 1)
axp.plot(t, rmse, '.-', label='running MSE')
axp.plot(t, se, '.-', label='SE')
'''
ylbl = ['$x$', '$y$', '$\\theta$', '$\\dot{x}$', '$\\dot{y}$',
        '$\\dot{\\theta}$']
for i in range(len(Xt.T) - 2):
    if i == 2:
        continue
    pe = 100 * (out[:, i] - estmean[:, i]) / out[:, i]
    axp.plot(t, pe, '.-', label=ylbl[i])
axp.plot(t, 100 * (dtrue - dest) / dtrue, '.-', label='body-ctrd dist')
axp.legend(loc='upper right')
axp.set_xlabel('$t$')
axp.set_title(axp.get_figure().get_label())
axp.set_ylabel('percent error')
'''

ipychk()
