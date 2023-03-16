__all__ = ['r2d', 'newplot', 'ipychk', 'gen_transform']
"""
Helper functions to keep main.py clean and readable
"""

import matplotlib.pyplot as plt
from numpy import *


def newplot(num=None):
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
    return array([[cos(t), -sin(t)], [sin(t), cos(t)]])


def ipychk():
    try:
        get_ipython()
        plt.ion()
    except NameError:
        plt.ioff()
        plt.show()


def gen_transform(r, th):
    R = r2d(th)
    xy = R @ [r, 0.0]
    out = c_[R, xy]
    out = r_[out, [[0, 0, 1]]]
    return out
