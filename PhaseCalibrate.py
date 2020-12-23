# -*- coding: UTF-8 -*-
import numpy as np


def phaseCalibration(csi, subCarrierIndex, rxNum, subCarrierNum):

    phaseRaw = np.angle(csi)
    phaseUnwrapped = np.unwrap(phaseRaw)

    for antIndexForPhase in range(1, rxNum):
        if phaseUnwrapped[antIndexForPhase, 0] - phaseUnwrapped[0, 0] > np.pi:
            phaseUnwrapped[antIndexForPhase, :] -= 2 * np.pi
        elif phaseUnwrapped[antIndexForPhase, 0] - phaseUnwrapped[0, 0] < -np.pi:
            phaseUnwrapped[antIndexForPhase, :] += 2 * np.pi

    phase = phaseUnwrapped.reshape(-1)
    a_mat = np.tile(subCarrierIndex, (1, rxNum))
    a_mat = np.append(a_mat, np.ones((1, subCarrierNum * rxNum)), axis=0)
    a_mat = a_mat.transpose((1, 0))
    a_mat_inv = np.linalg.pinv(a_mat)
    x = np.dot(a_mat_inv, phase)
    phaseSlope = x[0]
    phaseCons = x[1]
    calibration = np.exp(1j * (-phaseSlope * np.tile(subCarrierIndex, rxNum).reshape(3, -1) - phaseCons * np.ones((rxNum, subCarrierNum))))
    csi = csi*calibration

    return csi
