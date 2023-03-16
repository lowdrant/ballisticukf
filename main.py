#!/usr/bin/env python3
"""
Run simulation and estimation

Choose units of mass, length, time s.t.

m = r = g = 1

So motion becomes a function of the ratios b/t initial velocity to g
"""
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import rand, seed
from scipy.integrate import odeint

from helpers import *

dt = 0.01
t1 = 1
npts = 4

q0 = (0, 0, 0)   # x,y,theta
xi0 = (0, 0, 2)  # xdot,ydot,thetadot

# ===================================================================
# Generate Points for Observation

seed(0)
tf = []
for i in range(npts):
    r = rand()
    th = 2 * pi * rand()
    tf.append(gen_transform(r, th))

# ===================================================================
# Compute Motion


def ode(x, t):
    x = asarray(x)
    xdot = zeros_like(x)
    N = len(x)
    xdot[:N // 2] = x[N // 2:]
    xdot[N // 2:] = [0, -1, 0]
    return xdot


# Dynamics
x0 = r_[q0, xi0]
t = arange(0.0, t1, dt)
out = odeint(ode, x0, t)
xcom, ycom, thcom = out[:, 0], out[:, 1], out[:, 2]

# RBTs
gcom = zeros((3, 3, len(out)))
gcom[:-1, :-1] = r2d(thcom)
gcom[0, -1] = xcom
gcom[1, -1] = ycom
gcom[-1, -1] = 1
gp = einsum('ijk,njm->imnk', gcom, tf)

# ===================================================================
# Verification

for i in range(npts):
    d2 = (gcom[0, -1] - gp[0, -1, i])**2 + (gcom[1, -1] - gp[1, -1, i])**2
    assert sum(abs(diff(d2))) < 1e-6, 'rigid body assumption violated!'

# ===================================================================
# Estimation

# ===================================================================
# Plot

axp = newplot('parametric motion')
axp.plot(xcom, ycom, label='CoM')
for i in range(npts):
    axp.plot(gp[0, -1, i], gp[1, -1, i], label=f'pt{i+1}')
axp.legend()
axp.set_xlabel('x')
axp.set_ylabel('y')
axp.set_aspect('equal')

ipychk()
