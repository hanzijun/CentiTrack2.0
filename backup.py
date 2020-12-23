# -*- coding: UTF-8 -*-
import numpy as np
import math
from scipy.optimize import root, fsolve
THRESHOLD = 5.5


def get_Real(subcarrier):
    return map(lambda x: float("%.4f" % x.real), subcarrier)


def get_Imag(subcarrier):
    return map(lambda x: float("%.4f" % x.imag), subcarrier)


def rotate(x, y, theta):
    x_r = x * np.cos(theta) + y * np.sin(theta)
    y_r = y * np.cos(theta) - x * np.sin(theta)
    return x_r, y_r


def mn_to_xy_shitty(m, n):  # TODO: should be revised for 3D tracking
    sqrt = 2*m**2-3*m**2*n**2+m**2*n**4+2*m**3*n-2*m**3*n**3-3*m**4+4*m**4*n**2-m**4*n**4-2*m**5*n-m**6*n**2+2*m**5*n**3+m**6
    if sqrt >= 0:
        x = (-1+m**2-m*n+n**2-(m*n)**2+m*n**3+math.sqrt(sqrt))/(2*(-1+m**2+n**2))
        y = (m-n+m**2*n-m*n**2+(-n+m**2*n-m*n**2+n**3-m**2*n**3+m*n**4+n*math.sqrt(sqrt))/(-1+m**2+n**2))/(2*m)
    else:
        x = (-1 + m ** 2 - m * n + n ** 2 - (m * n) ** 2 + m * n ** 3) / (2 * (-1 + m ** 2 + n ** 2))
        y = (m - n + m ** 2 * n - m * n ** 2 + (-n + m ** 2 * n - m * n ** 2 + n ** 3 - m ** 2 * n ** 3 + m * n ** 4) / (-1 + m ** 2 + n ** 2)) / (2 * m)
    return x, y


def xyz_3d_coordinates2(p, q, aoa1, aoa2):
    a, b = aoa1, aoa2
    x = (p**2*q - p*q**2 - 2*np.sqrt(2)*a*p*q + p + np.sqrt(2)*a*q**2 + q - np.sqrt(2)*a)/(2*(p + q - np.sqrt(2)*a))
    y = (- p**2*q - np.sqrt(2)*a*p**2 + p*q**2 + p + q - np.sqrt(2)*a)/(2*(p + q - np.sqrt(2)*a))
    z = (-(2*a**2*p**4 + 8*a**2*p**2*q**2 - 4*a**2*p**2 - 8*a**2*p*q**3 + 8*a**2*p*q + 2*a**2*q**4 - 4*a**2*q**2
           + 4*a**2 + 2*np.sqrt(2)*a*p**4*q - 6*np.sqrt(2)*a*p**3*q**2 + 2*np.sqrt(2)*a*p**3 + 6*np.sqrt(2)*a*p**2*q**3
           - 6*np.sqrt(2)*a*p**2*q - 2*np.sqrt(2)*a*p*q**4 + 2*np.sqrt(2)*a*p*q**2 - 4*np.sqrt(2)*a*p + 2*np.sqrt(2)*a*q**3
           - 4*np.sqrt(2)*a*q + 2*p**4*q**2 - p**4 - 4*p**3*q**3 + 2*p**2*q**4 - 2*p**2*q**2 + 2*p**2 + 4*p*q - q**4
           + 2*q**2)/(2*a**2 - 2*np.sqrt(2)*a*p - 2*np.sqrt(2)*a*q + p**2 + 2*p*q + q**2))**0.5/2
    return x, y, z


def xyz_3d_coordinates3(p, q, aoa1, aoa2):
    a, b = aoa1, aoa2
    x = (p**2*q - p*q**2 + p - np.sqrt(2)*b*q**2 + q - np.sqrt(2)*b)/(2*(p + q - np.sqrt(2)*b))
    y = (- p**2*q + np.sqrt(2)*b*p**2 + p*q**2 - 2*np.sqrt(2)*b*p*q + p + q - np.sqrt(2)*b)/(2*(p + q - np.sqrt(2)*b))

    z = (-(2*b**2*p**4 - 8*b**2*p**3*q + 8*b**2*p**2*q**2 - 4*b**2*p**2 + 8*b**2*p*q + 2*b**2*q**4 - 4*b**2*q**2
           + 4*b**2 - 2*np.sqrt(2)*b*p**4*q + 6*np.sqrt(2)*b*p**3*q**2 + 2*np.sqrt(2)*b*p**3 - 6*np.sqrt(2)*b*p**2*q**3
           + 2*np.sqrt(2)*b*p**2*q + 2*np.sqrt(2)*b*p*q**4 - 6*np.sqrt(2)*b*p*q**2 - 4*np.sqrt(2)*b*p + 2*np.sqrt(2)*b*q**3
           - 4*np.sqrt(2)*b*q + 2*p**4*q**2 - p**4 - 4*p**3*q**3 + 2*p**2*q**4 - 2*p**2*q**2 + 2*p**2 + 4*p*q - q**4
           + 2*q**2)/(2*b**2 - 2*np.sqrt(2)*b*p - 2*np.sqrt(2)*b*q + p**2 + 2*p*q + q**2))**0.5/2

    return x, y, z


def xy2z(x, y, p, q):
    z = (q**4 - 4*q**2*x**2 + 4*q**2*x - 4*q**2*y**2 - 2*q**2 + 4*x**2 - 4*x + 1)**0.5 / (2*q)
    # z2 = (p**4 - 4*p**2*x**2 + 4*p**2*y - 4*p**2*y**2 - 2*p**2 + 4*y**2 - 4*y + 1)**0.5 / (2*p)
    return z


def mn_to_xy_series(ns, ms, aoa1, aoa2):
    msc, nsc = np.array(ms), np.array(ns)
    msc, nsc = 1.6-(msc - msc[0])/100, 1.6-(nsc - nsc[0])/100
    trace_x1, trace_y1 = np.zeros(msc.shape), np.zeros(msc.shape)
    trace_x2, trace_y2 = np.zeros(msc.shape), np.zeros(msc.shape)
    trace_z1, trace_z2, trace_z3 = np.zeros(msc.shape), np.zeros(msc.shape), np.zeros(msc.shape)

    for i in range(0, len(ns)):
        m_in, n_in = max(1, msc[i]), max(1, nsc[i])
        aoa1[i], aoa2[i] = np.cos(aoa1[i] / 180 * np.pi), np.cos(aoa2[i] / 180 * np.pi)
        x1, y1, z1 = xyz_3d_coordinates2(m_in, n_in, aoa1[i], aoa2[i])
        x2, y2, z2 = xyz_3d_coordinates3(m_in, n_in, aoa1[i], aoa2[i])
        z = xy2z((x1 + x2) / 2, (y1 + y2) / 2, m_in, n_in)

        # print "p:   ", m_in, "  q:   ", n_in, "  a:   ", aoa1[i], "   b:   ", aoa2[i]
        # print "x1:   ", x1, "  y1:   ", y1, "    z1: ", z1
        # print "x2:   ", x2, "  y2:   ", y2, "    z2: ", z2
        x1, y1 = 100*x1, 100*y1
        x2, y2 = 100*x2, 100*y2
        trace_z1[i] = 100 * z1
        trace_z2[i] = 100 * z2
        trace_z3[i] = 100 * z

        trace_x1[i], trace_y1[i] = rotate(x1, y1, -0.25 * np.pi)
        trace_x2[i], trace_y2[i] = rotate(x2, y2, -0.25 * np.pi)

    trace_x1, trace_y1 = trace_x1 - trace_x1[0], trace_y1 - trace_y1[0]
    trace_x2, trace_y2 = trace_x2 - trace_x2[0], trace_y2 - trace_y2[0]

    return [trace_x1, trace_y1, trace_z1], [trace_x2, trace_y2, trace_z2], trace_z3


def find_peaks_(phase_list):
    peak_segments = []
    peaks = []
    p = []
    flag = 0
    for index in range(1, len(phase_list)-2):
        if abs(phase_list[index] - phase_list[index+1]) > THRESHOLD / 2.0 > abs(phase_list[index] - phase_list[index - 1]) and abs(phase_list[index + 1] - phase_list[index + 2]) < THRESHOLD / 2.0:
            p.append(index)
            if phase_list[index] > phase_list[index+1]:
                if flag == 0:
                    peaks.append(index)
                else:
                    peak_segments.append(peaks)
                    peaks = [index]
                flag = 0
            else:
                if flag == 1:
                    peaks.append(index)
                    flag = 1
                else:
                    if peaks:
                        peak_segments.append(peaks)
                        peaks = [index]
                    else:
                        peaks.append(index)
                    flag = 1
    peak_segments.append(peaks)
    return p, peak_segments


def calibration(a):
    phase = np.angle(a)
    gaps, ps = find_peaks_(phase)
    ps = filter(lambda x: len(x) > 0, ps)
    angle_calibrated = np.zeros(np.shape(phase), dtype=float)
    parts = np.zeros(np.shape(a), dtype=complex)

    if len(ps) == 0:
        angle_calibrated = phase
        parts = a
    elif len(ps) == 1:
        if len(ps[0]) == 1:
            angle_calibrated = phase
            parts = a
        else:
            for i in range(0, gaps.__len__() - 1):
                angle_calibrated[gaps[i]:gaps[i + 1]] = np.angle(a[gaps[i]:gaps[i + 1]] - np.mean(a[gaps[i]:gaps[i + 1]]))
                parts[gaps[i]:gaps[i + 1]] = a[gaps[i]:gaps[i + 1]] - np.mean(a[gaps[i]:gaps[i + 1]])
            angle_calibrated[0:gaps[0]] = np.angle(a[0:gaps[0]] - np.mean(a[gaps[0]:gaps[1]]))
            angle_calibrated[gaps[-1]:] = np.angle(a[gaps[-1]:] - np.mean(a[gaps[-2]:gaps[-1]]))
            parts[0:gaps[0]] = a[0:gaps[0]] - np.mean(a[gaps[0]:gaps[1]])
            parts[gaps[-1]:] = a[gaps[-1]:] - np.mean(a[gaps[-2]:gaps[-1]])
    else:
        if len(ps[0]) == 1:
            angle_calibrated[:(ps[0][0]+ps[1][0])/2] = phase[:(ps[0][0]+ps[1][0])/2]
            parts[:(ps[0][0]+ps[1][0])/2] = a[:(ps[0][0]+ps[1][0])/2]
        else:
            ps_l = (ps[0][-1] - ps[0][0]) / (ps[0].__len__() - 1)
            for i in range(len(ps[0]) - 1):
                angle_calibrated[ps[0][i]: ps[0][i + 1]] = np.angle(a[ps[0][i]:ps[0][i + 1]] - np.mean(a[ps[0][i]:ps[0][i + 1]]))
                parts[ps[0][i]: ps[0][i + 1]] = a[ps[0][i]:ps[0][i + 1]] - np.mean(a[ps[0][i]:ps[0][i + 1]])

            if ps[0][0] - ps_l <= 0:
                angle_calibrated[0:ps[0][0]] = np.angle(a[0:ps[0][0]] - np.mean(a[ps[0][0]:ps[0][1]]))
                parts[0:ps[0][0]] = a[0:ps[0][0]] - np.mean(a[ps[0][0]:ps[0][1]])
            else:
                angle_calibrated[ps[0][0] - ps_l: ps[0][0]] = np.angle(a[ps[0][0] - ps_l: ps[0][0]] - np.mean(a[ps[0][0]:ps[0][1]]))
                angle_calibrated[0:ps[0][0] - ps_l] = [i+angle_calibrated[ps[0][0]-ps_l]-phase[ps[0][0]-ps_l-1] for i in phase[0: ps[0][0] - ps_l]]
                parts[ps[0][0] - ps_l: ps[0][0]] = a[ps[0][0] - ps_l: ps[0][0]] - np.mean(a[ps[0][0]:ps[0][1]])
                parts[0:ps[0][0] - ps_l] = [i + parts[ps[0][0] - ps_l] - a[ps[0][0] - ps_l - 1] for i in a[0: ps[0][0] - ps_l]]

            angle_calibrated[ps[0][-1]: (ps[0][-1] + ps[1][0])/2] = np.angle(a[ps[0][-1]: (ps[0][-1] + ps[1][0]) / 2] - np.mean(a[ps[0][-2]:ps[0][-1]]))
            parts[ps[0][-1]: (ps[0][-1] + ps[1][0])/2] = a[ps[0][-1]: (ps[0][-1] + ps[1][0]) / 2] - np.mean(a[ps[0][-2]:ps[0][-1]])

        for peaks in range(1, len(ps) - 1):
            if len(ps[peaks]) == 1:
                angle_calibrated[(ps[peaks-1][-1]+ps[peaks][0])/2: (ps[peaks][0]+ps[peaks+1][0])/2] = phase[(ps[peaks-1][-1]+ps[peaks][0])/2: (ps[peaks][0]+ps[peaks+1][0])/2]
                parts[(ps[peaks-1][-1]+ps[peaks][0])/2: (ps[peaks][0]+ps[peaks+1][0])/2] = a[(ps[peaks-1][-1]+ps[peaks][0])/2: (ps[peaks][0]+ps[peaks+1][0])/2]
            else:
                for i in range(len(ps[peaks]) - 1):
                    angle_calibrated[ps[peaks][i]: ps[peaks][i+1]] = np.angle(a[ps[peaks][i]:ps[peaks][i+1]]-np.mean(a[ps[peaks][i]:ps[peaks][i + 1]]))
                    parts[ps[peaks][i]: ps[peaks][i+1]] = a[ps[peaks][i]:ps[peaks][i+1]]-np.mean(a[ps[peaks][i]:ps[peaks][i + 1]])
                angle_calibrated[(ps[peaks-1][-1] + ps[peaks][0])/2: ps[peaks][0]] = np.angle(a[(ps[peaks-1][-1] + ps[peaks][0]) / 2: ps[peaks][0]] - np.mean(a[ps[peaks][0]:ps[peaks][1]]))
                angle_calibrated[ps[peaks][-1]: (ps[peaks][-1]+ps[peaks+1][0])/2] = np.angle(a[ps[peaks][-1]: (ps[peaks][-1]+ps[peaks + 1][0]) / 2] - np.mean(a[ps[peaks][-2]:ps[peaks][-1]]))
                parts[(ps[peaks-1][-1] + ps[peaks][0])/2: ps[peaks][0]] = a[(ps[peaks-1][-1] + ps[peaks][0]) / 2: ps[peaks][0]] - np.mean(a[ps[peaks][0]:ps[peaks][1]])
                parts[ps[peaks][-1]: (ps[peaks][-1]+ps[peaks+1][0])/2] = a[ps[peaks][-1]: (ps[peaks][-1]+ps[peaks + 1][0]) / 2] - np.mean(a[ps[peaks][-2]:ps[peaks][-1]])
        if len(ps[-1]) == 1:
            angle_calibrated[(ps[-2][-1]+ps[-1][0])/2:] = [i+angle_calibrated[(ps[-2][-1]+ps[-1][0])/2-1]-phase[(ps[-2][-1]+ps[-1][0])/2] for i in phase[(ps[-2][-1]+ps[-1][0])/2:]]
            parts[(ps[-2][-1]+ps[-1][0])/2:] = [i+parts[(ps[-2][-1]+ps[-1][0])/2-1]-a[(ps[-2][-1]+ps[-1][0])/2] for i in a[(ps[-2][-1]+ps[-1][0])/2:]]

        else:
            ps_r = (ps[-1][-1] - ps[-1][0]) / (ps[-1].__len__() - 1)
            for i in range(len(ps[-1]) - 1):
                angle_calibrated[ps[-1][i]: ps[-1][i + 1]] = np.angle(a[ps[-1][i]:ps[-1][i + 1]] - np.mean(a[ps[-1][i]:ps[-1][i + 1]]))
                parts[ps[-1][i]: ps[-1][i + 1]] = a[ps[-1][i]:ps[-1][i + 1]] - np.mean(a[ps[-1][i]:ps[-1][i + 1]])
            if ps[-1][-1] + ps_r >= 500:
                angle_calibrated[ps[-1][-1]:] = np.angle(a[ps[-1][-1]:] - np.mean(a[ps[-1][-2]:ps[-1][-1]]))
                parts[ps[-1][-1]:] = a[ps[-1][-1]:] - np.mean(a[ps[-1][-2]:ps[-1][-1]])
            else:
                angle_calibrated[ps[-1][-1]:ps[-1][-1] + ps_r] = np.angle(a[ps[-1][-1]:ps[-1][-1] + ps_r] - np.mean(a[ps[-1][-2]:ps[-1][-1]]))
                angle_calibrated[ps[-1][-1] + ps_r:] = [i+angle_calibrated[ps[-1][-1] + ps_r-1]-phase[ps[-1][-1]+ps_r] for i in phase[ps[-1][-1] + ps_r:]]
                parts[ps[-1][-1]:ps[-1][-1] + ps_r] = a[ps[-1][-1]:ps[-1][-1] + ps_r] - np.mean(a[ps[-1][-2]:ps[-1][-1]])
                parts[ps[-1][-1] + ps_r:] = [i+parts[ps[-1][-1] + ps_r-1]-a[ps[-1][-1]+ps_r] for i in a[ps[-1][-1] + ps_r:]]

            angle_calibrated[(ps[-2][-1] + ps[-1][0])/2:ps[-1][0]] = np.angle(a[(ps[-2][-1] + ps[-1][0]) / 2:ps[-1][0]] - np.mean(a[ps[-1][-2]:ps[-1][-1]]))
            parts[(ps[-2][-1] + ps[-1][0])/2:ps[-1][0]] = a[(ps[-2][-1] + ps[-1][0]) / 2:ps[-1][0]] - np.mean(a[ps[-1][-2]:ps[-1][-1]])

    return angle_calibrated, parts
