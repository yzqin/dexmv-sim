# trj, psg = min_jerk(pos, dur, vel, acc, psg)
#
# Compute minimum-jerk trajectory through specified points
#
# INPUTS:
# pos: NxD array with the D-dimensional coordinates of N points
# dur: number of time steps (integer)
# vel: 2xD array with endpoint velocities, [] sets vel to 0
# acc: 2xD array with endpoint accelerations, [] sets acc to 0
# psg: (N-1)x1 array of via-point passage times (between 0 and dur);
#      [] causes optimization over the passage times
#
# OUTPUTS
# trj: dur x D array with the minimum-jerk trajectory
# psg: (N-1)x1 array of passage times
#
# This is an implementation of the algorithm described in:
#  Todorov, E. and Jordan, M. (1998) Smoothness maximization along
#  a predefined path accurately predicts the speed profiles of
#  complex arm movements. Journal of Neurophysiology 80(2): 696-714
# The paper is available online at www.cogsci.ucsd.edu/~todorov

# Copyright (C) Emanuel Todorov, 1998-2006
# Python implementation by Dogancan Kebude
# https://github.com/dkebude/py-min-jerk/blob/master/min_jerk.py


import math
import numpy as np
import scipy.optimize
from numpy.linalg import inv


def min_jerk(pos=None, dur=None, vel=None, acc=None, psg=None):
    N = pos.shape[0]  # number of point
    D = pos.shape[1]  # dimensionality

    if not vel:
        vel = np.zeros((2, D))  # default endpoint vel is 0
    if not acc:
        acc = np.zeros((2, D))  # default endpoint acc is 0

    t0 = np.array([[0], [dur]])

    if psg is None:  # passage times unknown, optimize
        if N > 2:
            psg = np.arange(dur / (N - 1), dur - dur / (N - 1) + 1, dur / (N - 1)).T
            func = lambda psg_: mjCOST(psg_, pos, vel, acc, t0)
            psg = scipy.optimize.fmin(func=func, x0=psg)
        else:
            psg = []

    trj, vv, aa = mjTRJ(psg, pos, vel, acc, t0, dur)
    return trj, psg, vv, aa


################################################################
###### Compute jerk cost
################################################################

def mjCOST(t, x, v0, a0, t0):
    N = max(x.shape)
    D = min(x.shape)

    v, a = mjVelAcc(t, x, v0, a0, t0)
    aa = np.concatenate(([a0[0][:]], a, [a0[1][:]]), axis=0)
    aa0 = aa[0:N - 1][:]
    aa1 = aa[1:N][:]
    vv = np.concatenate(([v0[0][:]], v, [v0[1][:]]), axis=0)
    vv0 = vv[0:N - 1][:]
    vv1 = vv[1:N][:]
    tt = np.concatenate((t0[0], t, t0[1]), axis=0)
    T = np.diff(tt)[np.newaxis].T * np.ones((1, D))
    xx0 = x[0:N - 1][:]
    xx1 = x[1:N][:]

    j = 3 * (3 * aa0 ** 2 * T ** 4 - 2 * aa0 * aa1 * T ** 4 + 3 * aa1 ** 2 * T ** 4 + 24 * aa0 * T ** 3 * vv0 - \
             16 * aa1 * T ** 3 * vv0 + 64 * T ** 2 * vv0 ** 2 + 16 * aa0 * T ** 3 * vv1 - \
             24 * aa1 * T ** 3 * vv1 + 112 * T ** 2 * vv0 * vv1 + 64 * T ** 2 * vv1 ** 2 + \
             40 * aa0 * T ** 2 * xx0 - 40 * aa1 * T ** 2 * xx0 + 240 * T * vv0 * xx0 + \
             240 * T * vv1 * xx0 + 240 * xx0 ** 2 - 40 * aa0 * T ** 2 * xx1 + 40 * aa1 * T ** 2 * xx1 - \
             240 * T * vv0 * xx1 - 240 * T * vv1 * xx1 - 480 * xx0 * xx1 + 240 * xx1 ** 2) / T ** 5

    J = sum(sum(abs(j)));

    return J


################################################################
###### Compute trajectory
################################################################

def mjTRJ(tx, x, v0, a0, t0, P):
    N = max(x.shape)
    D = min(x.shape)
    X_list = []
    V_list = []
    A_list = []

    if len(tx) > 0:
        v, a = mjVelAcc(tx, x, v0, a0, t0)
        aa = np.concatenate(([a0[0][:]], a, [a0[1][:]]), axis=0)
        vv = np.concatenate(([v0[0][:]], v, [v0[1][:]]), axis=0)
        tt = np.concatenate((t0[0], tx, t0[1]), axis=0)
    else:
        aa = a0
        vv = v0
        tt = t0

    ii = 0
    for i in range(1, int(P) + 1):
        t = (i - 1) / (P - 1) * (t0[1] - t0[0]) + t0[0]
        if t > tt[ii + 1]:
            ii = ii + 1
        T = (tt[ii + 1] - tt[ii]) * np.ones((1, D))
        margin = t - tt[ii]
        t = (t - tt[ii]) * np.ones((1, D))
        aa0 = aa[ii][:]
        aa1 = aa[ii + 1][:]
        vv0 = vv[ii][:]
        vv1 = vv[ii + 1][:]
        xx0 = x[ii][:]
        xx1 = x[ii + 1][:]

        tmp = aa0 * t ** 2 / 2 + t * vv0 + xx0 + t ** 4 * (3 * aa0 * T ** 2 / 2 - aa1 * T ** 2 + \
                                                           8 * T * vv0 + 7 * T * vv1 + 15 * xx0 - 15 * xx1) / T ** 4 + \
              t ** 5 * (-(aa0 * T ** 2) / 2 + aa1 * T ** 2 / 2 - 3 * T * vv0 - 3 * T * vv1 - 6 * xx0 + \
                        6 * xx1) / T ** 5 + t ** 3 * (-3 * aa0 * T ** 2 / 2 + aa1 * T ** 2 / 2 - 6 * T * vv0 - \
                                                      4 * T * vv1 - 10 * xx0 + 10 * xx1) / T ** 3
        X_list.append(tmp[0])
        weight = margin / T[0, 0]
        V_list.append(vv0 * (1 - weight) + vv1 * weight)
        A_list.append(aa0 * (1 - weight) + aa1 * weight)

    return X_list, V_list, A_list


################################################################
###### Compute intermediate velocities and accelerations
################################################################

def mjVelAcc(t, x, v0, a0, t0):
    N = max(x.shape)
    D = min(x.shape)
    mat = np.zeros((2 * N - 4, 2 * N - 4))
    vec = np.zeros((2 * N - 4, D))
    tt = np.concatenate((t0[0], t, t0[1]), axis=0)

    for i in range(1, 2 * N - 4 + 1, 2):

        ii = int(math.ceil(i / 2.0))
        T0 = tt[ii] - tt[ii - 1]
        T1 = tt[ii + 1] - tt[ii]

        tmp = [-6 / T0, -48 / T0 ** 2, 18 * (1 / T0 + 1 / T1), \
               72 * (1 / T1 ** 2 - 1 / T0 ** 2), -6 / T1, 48 / T1 ** 2]

        if i == 1:
            le = 0
        else:
            le = -2

        if i == 2 * N - 5:
            ri = 1
        else:
            ri = 3

        mat[i - 1][i + le - 1:i + ri] = tmp[3 + le - 1:3 + ri]
        vec[i - 1][:] = 120 * (x[ii - 1][:] - x[ii][:]) / T0 ** 3 \
                        + 120 * (x[ii + 1][:] - x[ii][:]) / T1 ** 3

    for i in range(2, 2 * N - 4 + 1, 2):

        ii = int(math.ceil(i / 2.0))
        T0 = tt[ii] - tt[ii - 1]
        T1 = tt[ii + 1] - tt[ii]

        tmp = [48 / T0 ** 2, 336 / T0 ** 3, 72 * (1 / T1 ** 2 - 1 / T0 ** 2), \
               384 * (1 / T1 ** 3 + 1 / T0 ** 3), -48 / T1 ** 2, 336 / T1 ** 3]

        if i == 2:
            le = -1
        else:
            le = -3

        if i == 2 * N - 4:
            ri = 0
        else:
            ri = 2

        mat[i - 1][i + le - 1:i + ri] = tmp[4 + le - 1:4 + ri]
        vec[i - 1][:] = 720 * (x[ii][:] - x[ii - 1][:]) / T0 ** 4 \
                        + 720 * (x[ii + 1][:] - x[ii][:]) / T1 ** 4

    T0 = tt[1] - tt[0]
    T1 = tt[N - 1] - tt[N - 2]
    vec[0][:] = vec[0][:] + 6 / T0 * a0[0][:] + 48 / T0 ** 2 * v0[0][:]
    vec[1][:] = vec[1][:] - 48 / T0 ** 2 * a0[0][:] - 336 / T0 ** 3 * v0[0][:]
    vec[2 * N - 6][:] = vec[2 * N - 6][:] + 6 / T1 * a0[1][:] - 48 / T1 ** 2 * v0[1][:]
    vec[2 * N - 5][:] = vec[2 * N - 5][:] + 48 / T1 ** 2 * a0[1][:] - 336 / T1 ** 3 * v0[1][:]

    avav = inv(mat).dot(vec)
    a = avav[0:2 * N - 4:2][:]
    v = avav[1:2 * N - 4:2][:]

    return v, a


if __name__ == '__main__':
    position = np.stack([np.arange(12), np.cumsum(np.arange(12))], axis=1)
    psg = (np.arange(position.shape[0] - 2)) * 2 + 2
    trj, _, vel, acc = min_jerk(position, dur=24, psg=psg)
    print(trj)
    print(psg)
    print(vel)
    print(acc)
