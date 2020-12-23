# -*- coding=utf-8 -*-
import read_bf_file
from PCA import PCAtest
from PhaseCalibrate import phaseCalibration
import scipy.stats as sc
from skimage import feature
from scipy.signal import savgol_filter
from backup import *
from relative_motion_trace import trace

sampleFrequency = 100  # Hertz
centerFrequency = 5.32e9  # Hertz, 64 channel
speedOfLight = 299792458  # speed of electromagnetic wave
antDistance = 2.7e-2
rxAntennaNum = 3
subCarrierNum = 30
f_gap = 312.5e3
subCarrierIndex40 = np.array([-58, -54, -50, -46, -42, -38, -34, -30, -26, -22, -18, -14, -10, -6, -2,
                              2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58])
subCarrierIndex20 = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                              1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])


class Track(object):
    def __init__(self,
                 search_interval=(-np.pi / 2, np.pi/2),
                 toa_interval=(-2.5, 2.5),
                 slide_window=0.4,
                 filename=None,
                 use_mdl=1,
                 use_pca=1,
                 use40mhz=0,
                 use_trans=[5, 6],
                 step=20
                 ):
        self.search_interval = search_interval
        self.toa_interval = toa_interval
        self.slide_window = slide_window
        self.filename = filename
        self.useMDL = use_mdl
        self.usePCA = use_pca
        self.use40MHz = use40mhz
        self.useTrans = use_trans

        self.angleStepsNum = 400
        self.angleStepLen = (self.search_interval[1] - self.search_interval[0]) / self.angleStepsNum
        self.angleSteps = np.arange(self.search_interval[0], self.search_interval[1] + self.angleStepLen,
                                    self.angleStepLen, dtype=float)

        self.toaStepsNum = 400
        self.toaStepLen = (self.toa_interval[1] - self.toa_interval[0]) / self.toaStepsNum
        self.toaSteps = np.arange(self.toa_interval[0], self.toa_interval[1] + self.toaStepLen, self.toaStepLen,
                                  dtype=float)

        self.slideWindowLen = int(self.slide_window * sampleFrequency)
        self.stepLength = step
        self.overlapLength = self.slideWindowLen - self.stepLength
        file_r = read_bf_file.read_file(self.filename)
        self.aoa = self.readfile(file_r)

    def get_AoA(self):
        return self.aoa

    def topk(self, music_spectrum, max_p, k):
        max_v = [music_spectrum[p[0]][p[1]] for p in max_p]
        descend_index = np.argsort(max_v)
        return [max_p[i] for i in descend_index[-k:]]

    def getMUSIC(self, noiseMultiply, fc):
        angle_consider = self.angleSteps
        us_consider = (antDistance * fc / speedOfLight) * np.sin(angle_consider)
        delay_consider = 1e-7 * self.toaSteps
        subCarrierIndex = subCarrierIndex40 if self.use40MHz == 1 else subCarrierIndex20
        if self.usePCA:
            delay_steering_mat = np.exp(-1j * 2 * np.pi * subCarrierIndex[:self.useTrans[0]].reshape(self.useTrans[0], -1)
                                        * f_gap * delay_consider)
        else:
            delay_steering_mat = np.exp(-1j * 2 * np.pi * subCarrierIndex.reshape(30, -1) * f_gap * delay_consider)
        aoa_steering_mat = np.exp(-2j * np.pi * np.array([0, 1, 2]).reshape(3, 1) * us_consider)
        aoa_steering_inv_mat = 1
        theta_tau_mat = np.kron(aoa_steering_mat, delay_steering_mat)
        theta_tau_delta_mat = np.kron(aoa_steering_inv_mat, theta_tau_mat)  # =theta_tau_mat
        pna = np.dot(noiseMultiply, theta_tau_delta_mat)
        music_spectrum = np.sum(theta_tau_delta_mat.conjugate().transpose() * (pna.transpose()), axis=1)  # Ah*q*qh@At
        music_spectrum = 1 / np.abs(music_spectrum)
        music_spectrum = music_spectrum.reshape(401, -1)
        return music_spectrum

    def mdl_algorithm(self, eigenvalues):
        mdl = np.zeros(len(eigenvalues))
        lambda_tot = eigenvalues
        sub_arr_size = len(eigenvalues)
        n_segments = self.slideWindowLen
        max_multipath = len(lambda_tot)
        for k in range(0, max_multipath):
            mdl[k] = -n_segments * (sub_arr_size - k) * np.log10(sc.gmean(lambda_tot[k :]) / np.mean(lambda_tot[k:])) \
                     + 0.5 * k * (2 * sub_arr_size - k) * np.log10(n_segments)
        index = max(np.argmin(mdl), 1)
        print "source signals: ", index
        return index

    def getNoiseMat1(self, matrix):
        matr = np.zeros([90, 90], dtype=complex)
        for i in matrix:
            mat = np.asarray(i)[:, None]
            cor = np.dot(mat, mat.conjugate().transpose())
            matr += cor
        matr = matr / (len(matrix))
        eig, u_tmp = np.linalg.eig(matr)
        eig = np.abs(eig)
        un = np.argsort(-eig)
        eig = -np.sort(-eig)
        u = u_tmp[:, un[:]]
        if self.useMDL:
            index = self.mdl_algorithm(eig)
        else:
            index = 6
        qn = u[:, index:]
        noiseMultiply = np.dot(qn, qn.conjugate().transpose())
        return noiseMultiply, index

    def getNoiseMat(self, matrix) :
        mat = np.asarray(matrix).transpose()  # timestamp * Nrx
        cor = np.dot(mat, mat.conjugate().transpose())
        eig, u_tmp = np.linalg.eig(cor)
        eig = np.abs(eig)
        un = np.argsort(-eig)
        eig = -np.sort(-eig)
        u = u_tmp[:, un[:]]
        if self.useMDL:
            index = self.mdl_algorithm(eig)
        else :
            index = 6
        qn = u[:, index:]
        noiseMultiply = np.dot(qn, qn.conjugate().transpose())
        return noiseMultiply, index

    def format_matrix(self, csi):
        M, N = 3, 30
        if self.usePCA:
            ant1 = np.reshape(csi[0], (-1, self.useTrans[0])).transpose()
            ant2 = np.reshape(csi[1], (-1, self.useTrans[0])).transpose()
            ant3 = np.reshape(csi[2], (-1, self.useTrans[0])).transpose()
            csi_formed = np.concatenate((np.concatenate((ant1, ant2), axis=0), ant3), axis=0)
        else :
            csi_formed = np.zeros((M * N, 1), dtype=complex)
            for p in range(M):
                csi_formed = np.concatenate((np.concatenate((csi[0], csi[1]), axis=0), csi[2]), axis=0)

        return csi_formed

    def fillCSIMatrix(self, fileToRead):
        subCarrierIndex = subCarrierIndex40 if self.use40MHz == 1 else subCarrierIndex20
        CSIMatrix = np.zeros([len(fileToRead), rxAntennaNum, subCarrierNum], dtype=complex)
        if self.usePCA:
            CSIMatrixx = np.zeros([len(fileToRead), 3 * self.useTrans[0], self.useTrans[1]], dtype=complex)
        else:
            CSIMatrixx = np.zeros([len(fileToRead), subCarrierNum * 3], dtype=complex)
        timestampCount = 0
        for item in fileToRead:
            for EachCSI in range(0, 30):
                CSIMatrix[timestampCount, :, EachCSI] = \
                    np.array([item.csi[EachCSI, 0, 0], item.csi[EachCSI, 0, 1],
                              item.csi[EachCSI, 0, 2]])
            tmp = phaseCalibration(CSIMatrix[timestampCount], subCarrierIndex, rxNum=rxAntennaNum,
                                   subCarrierNum=subCarrierNum)
            CSIMatrixx[timestampCount] = self.format_matrix(tmp)
            timestampCount += 1
        return CSIMatrixx

    def getAoASpectrum(self, CSIMatrix):
        if self.usePCA == 1:
            MUSICSignalNum = []
            Qn, sourceNum = self.getNoiseMat(CSIMatrix)
            eachSpectrum = self.getMUSIC(Qn, centerFrequency)  # timestamp * Nrx
            MUSICSignalNum.append(sourceNum)
            AoASpectrum = np.array(eachSpectrum)
        else:
            MUSICSignalNum = []
            Qn, sourceNum = self.getNoiseMat1(CSIMatrix)
            eachSpectrum = self.getMUSIC(Qn, centerFrequency)  # timestamp * Nrx
            MUSICSignalNum.append(sourceNum)
            AoASpectrum = np.array(eachSpectrum)

        return AoASpectrum, sourceNum

    def readfile(self, *args):
        file1 = args[0]
        window_now = 0
        file_len = len(file1)
        c_matrix1 = np.zeros([self.slideWindowLen, subCarrierNum * 3], dtype=complex)
        AoAEstRx1, timeRx1 = [], None
        print("start timeStamp: ", str(file1[0].timestamp_low))
        while window_now + self.slideWindowLen < file_len:
            if window_now == 0:
                c_matrix1 = self.fillCSIMatrix(file1[0: self.slideWindowLen])
            else:
                c_matrix1[:self.overlapLength, :, :] = c_matrix1[-self.overlapLength:, :, :]
                c_matrix1[-self.stepLength:, :, :] = \
                    self.fillCSIMatrix(file1[window_now + self.overlapLength: window_now + self.slideWindowLen])
            aoa_tof = None
            if self.usePCA == 1:
                ret = np.zeros([c_matrix1.shape[0], c_matrix1.shape[1]], dtype=complex)
                for csi in range(len(c_matrix1)):
                    temp = PCAtest(c_matrix1[csi, :, :], 1)
                    ret[csi, :] = temp.reshape(-1)
                aoa_spectrum, peak_num = self.getAoASpectrum(ret)
                index_tem = imregionalmax(aoa_spectrum)
                topk_index = self.topk(aoa_spectrum, index_tem, peak_num)
            else:
                aoa_spectrum, peak_num = self.getAoASpectrum(c_matrix1)
                index_tem = imregionalmax(aoa_spectrum)
                topk_index = self.topk(aoa_spectrum, index_tem, peak_num)
            for ma in topk_index:
                aoa = (ma[0] * self.angleStepLen + self.search_interval[0]) * 180 / np.pi
                tof = (ma[1] * self.toaStepLen + self.toa_interval[0])
                new_aoa_tof = np.array([[aoa, tof]])
                aoa_tof = new_aoa_tof if aoa_tof is None else np.append(aoa_tof, new_aoa_tof, axis=0)
            AoAEstRx1.append(aoa_tof)
            window_now += self.stepLength

        return AoAEstRx1


def imregionalmax(spectrum):
    aa = feature.peak_local_max(spectrum, min_distance=15)   # 2D
    return aa


if __name__ == '__main__':

    slide_window = 0.4
    # The relative motion trace module
    tra1 = trace("./rx1.dat")
    tra2 = trace("./rx2.dat")
    share_len_trace = min(len(tra1), len(tra2))
    tra1 = tra1[:share_len_trace]
    tra2 = tra2[:share_len_trace]
    intercept_s = int(slide_window * sampleFrequency / 20)
    tra1 = tra1[::10][intercept_s:]
    tra2 = tra2[::10][intercept_s:]
    print "CSI trace len:   %d" % share_len_trace
    print "-------------------------------------------------------------------------------"

    rx1 = Track(search_interval=(-np.pi/2, np.pi/2), slide_window=slide_window, filename="./rx1.dat",
                use_pca=1, use_mdl=1, use40mhz=0, step=10)
    aoa1 = rx1.get_AoA()
    rx2 = Track(search_interval=(-np.pi/2, np.pi/2), slide_window=slide_window, filename="./rx2.dat",
                use_pca=1, use_mdl=1, use40mhz=0, step=10)
    aoa2 = rx2.get_AoA()
    print "-------------------------------------------------------------------------------"

    # The direct hand-reflected path can be extracted in aoa1/aoa2 with minimum ToF
    share_len_aoa = min(len(aoa1), len(aoa2))
    aoa1 = aoa1[:share_len_aoa]
    aoa2 = aoa2[:share_len_aoa]
    print "AoA len:   %d" % len(aoa1)

    total_len = min(share_len_trace, share_len_aoa)
    tra1 = tra1[:total_len]
    tra2 = tra2[:total_len]
    aoa1 = aoa1[:total_len]
    aoa2 = aoa2[:total_len]
    print "total len:   %d" % total_len

    # The 3D tracking model, prior to savgol-filter
    # Here the initial path length p and q (i.e., initial position) is assumed as 1.6 and 1.6.
    [x1, y1, z1], [x2, y2, z2], z3 = mn_to_xy_series(tra2, tra1, aoa2, aoa1)
    x1, y1 = savgol_filter(x1, 31, 3), savgol_filter(y1, 31, 3)
    x2, y2 = savgol_filter(x2, 31, 3), savgol_filter(y2, 31, 3)
