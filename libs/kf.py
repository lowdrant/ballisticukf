"""KF construction for main.py"""
__all__ = ['construct_kf']

from numpy import asarray, eye, zeros

from .ekf import G
from .EstimatorFactory import KFFactory
from .helpers import *


def construct_kf(M, N, dt, linpt):
    """
    INPUTS:
        M -- int -- observation space size
        N -- int -- state space size
        dt -- float -- time step
        linpt -- 5x1 -- state to linearize about
    """
    linpt = asarray(linpt)
    A = G(0, 0, linpt, dt)  # linearize about a point
    C = zeros((M, N))
    C[::2, 0] = 1
    C[1::2, 1] = 1
    C[:, 5:] = eye(M)  # C defined here since 'M' is determined by main.py
    B, D = 0, 0  # no input
    R = eye(N)
    Q = eye(M)
    return KFFactory(A, B, C, R, Q)
