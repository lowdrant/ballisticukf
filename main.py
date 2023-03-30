#!/usr/bin/env python3
"""
Run simulation and estimation. Makes use of try/excepts to skip already
completed computations in ipython debugging sessions

Choose units (mass, length, time) s.t. m = r = g = I = 1
"""
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import normal, rand, seed

from filters import *
from helpers import *

# Constants
dt = 0.04
t1 = 1
npts = 5
q0 = (0.1, -0.1, 0)   # x,y,theta
xi0 = (0.1, 0, 5)  # xdot,ydot,thetadot
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
# Filtering
# TODO: vectorize probability functions

obs = einsum('ij...,j->i...', gp, [0, 0, 1])
ctrd = obs[:-1].mean(1)  # maybe centroid will be easier to work with?


def pxt(xtm1):
    """generate probability cloud of current state given prev state"""
    loc = array(xtm1, copy=1)
    for i in range(3):
        loc[i] += dt * loc[3 + i]
    loc[4] -= dt
    scale = [0.1, 0.01, pi / 10, 0.01, 0.01, 1, 0.01, 0.01]
    return normal(loc=loc, scale=scale)


def pzt(zt, xt):
    """"Probability" that observations zt came from state xt. Implemented as a
    cost function of RBT prediction error of zt.

    NOTES:
        Since zt is a rigid point on body xt, augment xt with RBT to zt.
        The particle filter renormalizes the probability of the particles,
        so the output of this function doesn't need to cleanly integrate to 1.        
        This lets us return 1/(1+err), where `err` is the Euclidean distance
        between the observed marker and its predicted location by the RBTs.
        The form of the function makes low errors preferred while avoiding
        division by 0/numeric instability.
    """
    xt, zt = asarray(xt), asarray(zt)
    dx, dy = xt[-2:]  # marker in body frame
    x, y, th = xt[:3]  # body in world frame
    zt_hat = SE2(x, y, th) @ [dx, dy, 1]
    err = sum((zt - zt_hat[:-1])**2)
    return 1 / (1 + err)


pf = ParticleFilter(pxt, pzt, dbg=1)
Xt = zeros((len(t), M, 8))
Xt[-1] = [*ctrd[..., 0], 0, 0, 0, 1, 0.5, 0.5]
seed(0)
for i, _ in enumerate(t):
    Xt[i] = pf(Xt[i - 1], ctrd[..., i])

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
estgBC[0, -1] = estmean[:, -2]
estgBC[1, -1] = estmean[:, -1]

est_ctrd = einsum('ijk,jmk->imk', estgWB, estgBC)[[0, 1], -1]

dtrue = sqrt(sum((out[:, [0, 1]] - ctrd.T)**2, 1))
dest = sqrt(sum((estmean[:, [0, 1]] - est_ctrd.T)**2, 1))

# ===================================================================
# Plot

axp = newplot('parametric motion')
axp.grid(0)
axp.plot(gcom[0, -1], gcom[1, -1], label='CoM')
axp.plot(*ctrd, label='marker centroid')
axp.plot(*estmean.T[:2], '.-', label='estimated CoM')
axp.plot(*est_ctrd, '.-', label='estimated marker')
axp.legend(loc='lower left')
axp.set_xlabel('$x$')
axp.set_ylabel('$y$')
axp.set_aspect('equal')

num = 'filter output'
plt.figure(num).clf()
ylbl = ['$x$', '$y$', '$\\theta$', '$\\dot{x}$', '$\\dot{y}$', '$\\dot{\\theta}$',
        '$dx$', '$dy$']
_, axf = plt.subplots(nrows=Xt.shape[-1] - 1, sharex='all', num=num)
for i, ax in enumerate(axf[:-1]):
    ax.plot(t, out[:, i], '.-', label='ground truth')
    ax.plot(t, estmean[:, i], '.-', label='estimate')
    ax.set_ylabel(ylbl[i])
axf[-1].plot(t, dtrue, '.-', c='tab:blue')
axf[-1].plot(t, dest, '.-', c='tab:orange')
axf[-1].set_ylabel('dist')

num = 'pct err'
axp = newplot(num)
ylbl = ['$x$', '$y$', '$\\theta$', '$\\dot{x}$', '$\\dot{y}$',
        '$\\dot{\\theta}$', '$dx$', '$dy$']
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

for a in axf:
    a.grid()
#    a.legend(loc='lower left')
axf[-1].set_xlabel('$t$')
axf[0].set_title(a.get_figure().get_label())

ipychk()
