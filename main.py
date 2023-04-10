#!/usr/bin/env python3
"""
Run simulation and estimation.

Units (mass, length, time) are s.t. m = r = g = I = 1
"""
from argparse import ArgumentParser
from time import time

from numpy import arange, ceil, zeros
from numpy.random import seed

from ekf import construct_ekf
from filters import *
from helpers import *
from pf import construct_pf

# ============================================================================
# Numeric Parameters that are awkward to set via CLI


def _init_sigma(s):
    pass

# ============================================================================
# CLI


def _run_sim(args):
    """Parse CLI args and run disk sim.
    INPUTS:
        args -- output of parse.parse_args()
    """
    assert args.x0.count(',') == 4, '5 values needed to specify init. cond.'
    x0 = [float(v) for v in args.x0.replace(' ', '').split(',')]
    x0.insert(4, 0)
    t = arange(0, args.tf, args.dt)
    print('Running simulation...')
    _, simout, obs = compute_motion(x0, t, args.npts)
    print('Done!')
    return t, simout, obs


def _run_filter(args, obs):
    """Parse CLI args and run appropriate filter.
    INPUTS:
        args -- output of parse.parse_args()
    OUTPUTS:
        estimate -- LxN -- L estimates (one per time step) for N states
    """
    L = int(ceil(args.tf / args.dt))  # num timesteps in sim
    M = 2 * args.npts  # each marker has 2 quantities
    N = M + 5  # state vector is 5D, plus num marker quantities
    fstr = args.filter.lower()

    # Assemble Filtering Function
    filt = None
    if fstr == 'ekf':
        filt = construct_ekf(M, N, args.dt, args.njit)
    elif fstr == 'ukf':
        raise NotImplementedError('UKF tbf')
        filt = construct_ukf(M, N, args)
    elif fstr == 'kf':
        raise NotImplementedError('KF tbf')
        filt = construct_kf(M, N, args)
    elif fstr == 'pf':
        raise NotImplementedError('still CLI testing EKF')
        scale = [0.1, 0.1, 0.1, 0.1, 1] + [0.01] * M
        if args.scale is not None:
            scale = [float(v) for v in args.scale.replace(' ', '').split(',')]
        filt = construct_pf(M, N, args.dt, scale)
    else:
        raise RuntimeError('somehow selected invalid filter type')

    # Particle Filter has unique call signature
    if fstr == 'pf':
        P = args.particles
        X_t = zeros(L, P, N)
        print(f'Starting particle filter...')
        tref = time()
        seed(0)
        for i, _ in enumerate(obs.T):
            X_t[i] = filt(X_t[i - 1], obs.T[i].flatten())
        print(f'Done! t={time()-tref:.2f}s')
        return X_t.mean(0)

    # KFs all have same call signature
    mu_t = zeros((L, N))
    sigma_t = zeros((L, N, N))
    print(f'Starting {fstr.upper()}...')
    tref = time()
    seed(0)
    for i, _ in enumerate(obs.T):
        mu_t[i], sigma_t[i] = filt(
            mu_t[i - 1], sigma_t[i - 1], 0, obs.T[i].flatten())
    print(f'Done! t={time()-tref:.2f}s')
    return mu_t


parser = ArgumentParser()
parser.add_argument('--tf', default=5, type=float,
                    help='sim runtime; default:5')
parser.add_argument('--dt', default=0.1, type=float,
                    help='sim step time; default:0.1')
parser.add_argument('--npts', default=3, type=int,
                    help='number of markers on disk; default:3')
parser.add_argument('--filter', help='state estimator; default:ekf',
                    choices=('kf', 'ekf', 'ukf', 'pf'), default='ekf')
parser.add_argument('--x0', type=str, default='0,0,0,0,10',
                    help='sim IVP, comma-separated list: (x,y,vx,vy,w); default(0,0,0,0,10)')
parser.add_argument('--no-plot', action='store_true', help='suppress plotting')
parser.add_argument('--njit', action='store_true',
                    help='(EKF only) enable njit; default:False')
# parser.add_argument('--particles', type=int,
#                     help='number of particles for Particle Filter')
# parser.add_argument('--scale', type=str, default=None,
#                     help='(particle filter) resampling variances of state vector elements, comma-separated')
if __name__ == '__main__':
    args = parser.parse_args()
    t, simout, obs = _run_sim(args)
    est = _run_filter(args, obs)
    tru = reconstruct(est, simout, obs)
    if not args.no_plot:
        plots(t, tru, est)
