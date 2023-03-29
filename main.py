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
dt = 0.01
t1 = 1
npts = 4
q0 = (0, 0, 0)   # x,y,theta
xi0 = (0, 0, 2)  # xdot,ydot,thetadot

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
gcom = sim(x0, t)

# Point Motion
gp = einsum('ijk,njm->imnk', gcom, tf)

# Simple Validation
for i in range(npts):
    d2 = (gcom[0, -1] - gp[0, -1, i])**2 + (gcom[1, -1] - gp[1, -1, i])**2
    assert sum(abs(diff(d2))) < 1e-6, 'rigid body assumption violated!'

# ===================================================================
# Estimation

obs = einsum('ij...,j->i...', gp, [0, 0, 1])
ctrd = obs[:-1].mean(1)  # maybe centroid will be easier to work with?


def pxt(xtm1):
    xtm1 = asarray(xtm1)
    return normal(xtm1, [5] * len(xtm1))


def pzt(zt, xt):
    # TODO: come up with probability/pentalty func for observations
    return 0.5


pf = ParticleFilter(pxt, pzt)
Xt = zeros((len(t), 1000, 6))
for i, _ in enumerate(t):
    Xt[i] = pf(Xt[i - 1], ctrd[..., i])

# ===================================================================
# Plot

axp = newplot('parametric motion')
axp.plot(gcom[0, -1], gcom[1, -1], label='CoM', lw=3)
for i in range(npts):
    axp.plot(gp[0, -1, i], gp[1, -1, i], label=f'M{i+1}')
axp.plot(*ctrd, label='marker centroid', lw=3)
axp.legend(loc='lower left')
axp.set_xlabel('$x$')
axp.set_ylabel('$y$')
axp.set_aspect('equal')

num = 'filter output'
plt.figure(num).clf()
_, axf = plt.subplots(nrows=2, sharex='all', num=num)
c = ['tab:' + v for v in ('blue', 'orange')]
lbl = ['true ' + v for v in ('xc', 'yc')]
ylbl = [f'${v}$' for v in ('x', 'y')]
for i in range(2):
    axf[i].plot(t, Xt[:, i], '.-', label=f'Xt[{i}]')
    axf[i].plot(t, gcom[i, -1], '.-', c=c[i], label=lbl[i])
    axf[i].set_ylabel(ylbl[i])
for a in axf:
    a.grid()
    a.legend(loc='lower left')
axf[1].set_xlabel('$t$')
axf[0].set_title(a.get_figure().get_label())

ipychk()
