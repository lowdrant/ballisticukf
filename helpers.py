__all__ = ['r2d', 'newplot', 'ipychk', 'gen_transform', 'sim']
"""
Helper functions to keep main.py clean and readable
"""

import matplotlib.pyplot as plt
from numpy import *
from scipy.integrate import odeint


def newplot(num=None):
    """generate fresh figure with axis grid.
    INPUTS:
        num -- (optional) num argument for `plt.figure`
    OUTPUTS:
        ax -- axis for plotting
    """
    if num is None:
        f = plt.figure()
    else:
        f = plt.figure(num)
    f.clf()
    ax = f.add_subplot(111)
    ax.grid()
    ax.set_title(str(num))
    return ax


def r2d(t):
    """get 2d rotation matrix
    INPUTS:
        t -- Nx... angle, can be ndarray
    OUTPUTS:
        R -- 2x2xN... rotation matrices
    """
    return array([[cos(t), -sin(t)], [sin(t), cos(t)]])


def ipychk():
    """If ipython session, turn on interactive plots; else `plt.show()`."""
    try:
        get_ipython()
        plt.ion()
    except NameError:
        plt.ioff()
        plt.show()


def gen_transform(r, th):
    """Generate rigid body transform from polar coordinates
    INPUTS:
        r -- Nx... radial distance from reference frame
        th -- Nx... angle from reference frame x-axis
    OUTPUTS:
        g -- 3x3xN... SE(2) rigid body transfrom matrices
    """
    R = r2d(th)
    xy = R @ [r, 0.0]
    out = c_[R, xy]
    out = r_[out, [[0, 0, 1]]]
    return out


def SE2(x, y, th):
    """Generate SE(2) RBT from R^3 parameterization
    INPUTS:
        x -- x coord of transform
        y -- y coord of transform
        th -- angle of transform
    OUTPUTS:
        g -- 3x3 -- SE(2) RBT
    """
    out = zeros((3, 3), dtype=float)
    out[:2, :2] = r2d(th)
    out[0, 2] = x
    out[1, 2] = y
    out[2, 2] = 1
    return out


def _dynamics(x, t):
    """ballistic motion ode
    INPUTS:
        x -- 6xN... state vectors (x,y,theta,xdot,ydot,thetadot)
        t -- N... time values of state vectors. only included for odeint
    """
    x = asarray(x)
    xdot = zeros_like(x)
    N = len(x)
    xdot[:N // 2] = x[N // 2:]
    xdot[N // 2:] = [0, -1, 0]
    return xdot


def sim(x0, t, return_state=0):
    """Simulate ballistic motion
    INPUTS:
        x0 -- 6x1 initial state vector (x,y,theta,xdot,ydot,thetadot)
        t -- Nx1 vector of time instants for integration
        return_state -- (optional) if true, additionally return state vector
                        output of odeint
    OUTPUTS:
        gcom -- Nx3x3 motion of disk stored in SE(2) transformation matrices
        state -- Nx6 state vector output of odeint
    """
    out = odeint(_dynamics, x0, t)
    gcom = zeros((3, 3, len(out)), dtype=float)
    gcom[:-1, :-1] = r2d(out[:, 2])
    gcom[0, -1] = out[:, 0]
    gcom[1, -1] = out[:, 1]
    gcom[-1, -1] = 1
    if return_state:
        return gcom, out
    return gcom


def rms(x, axis=None):
    """Compute RMS of x
    INPUTS:
        x -- Nx... array of data
        axis -- (optional) axis along which to compute mean 
    OUTPUTS:
        rms -- sqrt(x.mean(axis))
    """
    x = asarray(x)
    return sqrt(x.mean(axis=axis))
