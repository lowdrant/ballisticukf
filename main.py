#!/usr/bin/env python3
"""
Run simulation and estimation. Makes use of try/excepts to skip already
completed computations in ipython debugging sessions

Choose units (mass, length, time) s.t. m = r = g = I = 1
"""
import matplotlib.pyplot as plt
from numpy import *
from numpy.random import rand, seed

from helpers import *

# Constants
dt = 0.01
t1 = 1
npts = 4
q0 = (0, 0, 0)   # x,y,theta
xi0 = (0, 0, 2)  # xdot,ydot,thetadot

# ===================================================================
# Boiler Plate

# Point Observations
try:
    assert len(tf) == npts
    print('Reusing point transforms')
except (NameError, AssertionError) as e:
    print(f'{e.args[0]}:\nComputing point transforms: npts={npts}')
    seed(0)
    tf = []
    for i in range(npts):
        r = rand()
        th = 2 * pi * rand()
        tf.append(gen_transform(r, th))
    print('Done!')

# Simulation
try:
    assert not RESIM and len(gcom.T) == len(t) and len(t) == ceil(t1 / dt)
    print('Reusing simulated data')
except (NameError, AssertionError) as e:
    x0 = r_[q0, xi0]
    print(f'{e.args[0]}:\nComputing simulation, x0={x0},tf={t1},dt={dt}...')
    t = arange(0, t1, dt)
    gcom = sim(x0, t)
    gp = einsum('ijk,njm->imnk', gcom, tf)
    print('Done!')
    RESIM = False

# Basic Data Validation
for i in range(npts):
    d2 = (gcom[0, -1] - gp[0, -1, i])**2 + (gcom[1, -1] - gp[1, -1, i])**2
    assert sum(abs(diff(d2))) < 1e-6, 'rigid body assumption violated!'

# ===================================================================
# Estimation

# ===================================================================
# Plot

axp = newplot('parametric motion')
axp.plot(gcom[0, -1], gcom[1, -1], label='CoM')
for i in range(npts):
    axp.plot(gp[0, -1, i], gp[1, -1, i], label=f'pt{i+1}')
axp.legend()
axp.set_xlabel('$x$')
axp.set_ylabel('$y$')
axp.set_aspect('equal')

ipychk()
