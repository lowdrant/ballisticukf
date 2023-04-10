"""
Helper functions to keep main.py clean and readable
"""
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS as TC
from numpy import (array, asarray, c_, cos, diff, einsum, isscalar, mean, pi,
                   prod, r_, sin, sqrt, sum, zeros, zeros_like)
from numpy.random import rand, seed
from scipy.integrate import odeint

__all__ = ['r2d', 'newplot', 'ipychk', 'gen_transform', 'sim', 'SE2', 'rms',
           'gen_markers', 'compute_motion', 'plots', 'reconstruct']
__all__ += ['plot_' + v for v in ('parametric', 'state', 'est', 'obs', 'rb')]

# ============================================================================
# Rigid Body Funcs


def r2d(t):
    """get 2d rotation matrix
    INPUTS:
        t -- Nx... angle, can be ndarray
    OUTPUTS:
        R -- 2x2xN... rotation matrices
    """
    return array([[cos(t), -sin(t)], [sin(t), cos(t)]])


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


def gen_markers(npts):
    """generate markers on disk for observations"""
    seed(0)
    tf = []
    for _ in range(npts):
        r = rand()
        th = 2 * pi * rand()
        tf.append(gen_transform(r, th))
    return asarray(tf)


def SE2(x, y, th):
    """Generate SE(2) RBT from R^3 parameterization
    INPUTS:
        x -- N... --  x coord of transform
        y -- N... -- y coord of transform
        th -- angle of transform
    OUTPUTS:
        g -- 3x3xN... -- SE(2) RBT
    """
    N = 1
    if not isscalar(x):
        N = len(x)
    out = zeros((3, 3, N), dtype=float).squeeze()
    out[:2, :2] = r2d(th)
    out[0, 2] = x
    out[1, 2] = y
    out[2, 2] = 1
    return out

# ============================================================================
# Dynamics Funcs


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


def compute_motion(x0, t, npts):
    # Body Motion
    gcom, out = sim(x0, t, return_state=1)

    # Point Motion
    tf = gen_markers(npts)
    gp = einsum('ijk,njm->imnk', gcom, tf)
    obs = einsum('ij...,j->i...', gp, [0, 0, 1])[:-1]

    # Simple Validation
    for i in range(npts):
        d2 = (gcom[0, -1] - gp[0, -1, i])**2 + (gcom[1, -1] - gp[1, -1, i])**2
        assert mean(abs(diff(d2))) < 1e-6, 'rigid body assumption violated!'

    return gcom, out, obs

# ============================================================================
# Plotting


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


def ipychk():
    """If ipython session, turn on interactive plots; else `plt.show()`."""
    try:
        get_ipython()
        plt.ion()
    except NameError:
        plt.ioff()
        plt.show()


def plot_parametric(t, tru, est, kwest, c, lbls):
    axp = newplot('parametric motion')
    axp.grid(0)
    axp.plot(*tru[:, [0, 1]].T, label='CoM', c='tab:blue')
    axp.plot(*est.T[:2], '.-', label='estimated CoM', c='tab:blue', **kwest)
    c = c[1:]
    for i in range(0, est.shape[1] - 5, 2):
        j, k = 5 + i, i // 2
        tt = tru[:, [j, j + 1]] + tru[:, [0, 1]]
        ee = est[:, [j, j + 1]] + est[:, [0, 1]]
        axp.plot(*tt.T, c=c[k])  # ,label=f'mkr{k}')
        axp.plot(*ee.T, '.', c=c[k], **kwest)
    axp.legend(loc='upper left')
    axp.set_xlabel('$x$')
    axp.set_ylabel('$y$')
    axp.set_aspect('equal')
    return axp


def plot_state(t, tru, est, kwest, c, lbls):
    num = 'state estimates'
    plt.figure(num).clf()
    _, axs = plt.subplots(nrows=5, sharex='all', num=num)
    for i, ax in enumerate(axs):
        ax.plot(t, tru[:, i], '--', lw=3)
        ax.plot(t, est[:, i], '.-')
        lbl = lbls[i] if i < 5 else f'm{chr(ord("x") + (i - 5) % 2)}{(i - 5) // 2}'
        ax.set_ylabel(lbl)  # , rotation=0)
    for a in axs:
        a.grid(1)
    axs[-1].set_xlabel('$t$')
    axs[0].set_title(axs[0].get_figure().get_label())
    return axs


def plot_obs(t, tru, est, kwest, c, lbls):
    num = 'marker position'
    plt.figure(num).clf()
    _, axm = plt.subplots(nrows=(tru.shape[1] - 5), sharex='all', num=num)
    for i, ax in enumerate(axm):
        k = 5 + i
        tt = tru[:, k] + tru[:, i % 2]
        ee = est[:, k] + est[:, i % 2]
        ax.plot(t, tt, '--', lw=3)
        ax.plot(t, ee, '.-')
        lbl = f'm{chr(ord("x") + i % 2)}{i // 2}'
        ax.set_ylabel(lbl)  # , rotation=0)
    for a in axm:
        a.grid(1)
    axm[-1].set_xlabel('$t$')
    axm[0].set_title(axm[0].get_figure().get_label())
    return axm


def plot_est(t, tru, est, kwest, c, lbls):
    num = 'marker delta'
    plt.figure(num).clf()
    _, axe = plt.subplots(nrows=tru.shape[1] - 5, sharex='all', num=num)
    for i, ax in enumerate(axe):
        j = 5 + i
        ax.plot(t, tru[:, j], '--', lw=3)
        ax.plot(t, est[:, j], '.-')
        lbl = f'm{chr(ord("x") + i % 2)}{i // 2}'
        ax.set_ylabel(lbl)  # , rotation=0)
    for a in axe:
        a.grid(1)
    axe[-1].set_xlabel('$t$')
    axe[0].set_title(axe[0].get_figure().get_label())
    return axe


def plot_rb(t, tru, est, kwest, c, lbls):
    ax = newplot('rb params')
    for i in range((est.shape[1] - 5) // 2):
        k = 5 + 2 * i
        rtru = sqrt(sum(tru[..., [k, k + 1]]**2, 1))
#        rout = sqrt(sum((out[..., [0, 1]] - est[..., i, :].T)**2, -1))
        rest = sqrt(sum((est[..., [k, k + 1]])**2, -1))
 #       ax.plot(t, rout, '--', c=c[i], lw=3)  # ,label=f'out {i})
        ax.plot(t, rtru, '.-', c=c[i], ms=2)  # ,label=f'true {i}')
        ax.plot(t, rest, 'x-', label=f'$\\hat{{r}}_{i}$', c=c[i])
    ax.legend(loc='upper left')
    return ax


def plots(t, tru, est):
    """Group plotting function"""
    kwest = {'ms': 2, 'lw': 0.5, 'alpha': 1}
    lbls = ['$x$', '$y$', '$\\dot{x}$', '$\\dot{y}$', '$\\dot{\\theta}$']
    c = list(TC.keys())
    args = [t, tru, est, kwest, c, lbls]

    axp = plot_parametric(*args)
    axs = plot_state(*args)
    axo = plot_obs(*args)
    axe = plot_est(*args)
    axr = plot_rb(*args)
    ipychk()
    return axp, axs, axo, axe, axr

# ============================================================================
# Numerics


def rms(x, axis=None):
    """Compute RMS of x
    INPUTS:
        x -- Nx... array of data
        axis -- (optional) axis along which to compute mean
    OUTPUTS:
        rms -- sqrt((x**2).mean(axis))
    """
    x = asarray(x)
    return sqrt((x**2).mean(axis=axis))


def reconstruct(est, out, obs):
    tru = zeros_like(est)
    tru[:, :5] = out[:, [0, 1, 3, 4, 5]].copy()  # 2 is theta; skip
    tru[:, 5:] = obs.T.reshape(len(obs.T),
                               prod([v for v in obs.T.shape[1:]])).copy()
    tru[:, 5::2] -= tru[:, [0]]
    tru[:, 6::2] -= tru[:, [1]]
    return tru
