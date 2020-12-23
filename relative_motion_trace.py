import read_bf_file
from scipy.signal import savgol_filter
import pylab
from backup import *

TIMEINYERVAL = 0.01
TIMEBIASE = 0.008
TIMELEN = 500
IMAGETOCSIRATIO = 2
THRESHOLD = 5.5


def csi_ratio(an1, an2, an3):
    ret1, ret2 = None, None
    for sub_index in range(len(an1)):
        ret1 = np.array([np.divide(an1[sub_index], an2[sub_index])]) if ret1 is None else np.append(ret1, [np.divide(an1[sub_index], an2[sub_index])], axis=0)
        ret2 = np.array([np.divide(an2[sub_index], an3[sub_index])]) if ret2 is None else np.append(ret2, [np.divide(an2[sub_index], an3[sub_index])], axis=0)

    return ret1, ret2


def trace(filepath):
    file = read_bf_file.read_file(filepath)
    file_len = len(file)
    timestamp = np.array([])
    startTime = file[0].timestamp_low
    print "Length of packets: ", file_len, "    Start timestamp:" + str(startTime)
    ap1_tx1, ap2_tx1, ap3_tx1 = [], [], []
    for item in file :
        timestamp = np.append(timestamp, (item.timestamp_low - startTime) / 1000000.0)
        for eachcsi in range(0, 30):
            ap1_tx1.append(item.csi[eachcsi][0][0])
            ap2_tx1.append(item.csi[eachcsi][0][1])
            ap3_tx1.append(item.csi[eachcsi][0][2])

    ap1_tx1 = np.reshape(ap1_tx1, (file_len, 30)).transpose()
    ap2_tx1 = np.reshape(ap2_tx1, (file_len, 30)).transpose()
    ap3_tx1 = np.reshape(ap3_tx1, (file_len, 30)).transpose()

    ret1, ret2 = csi_ratio(ap1_tx1, ap2_tx1, ap3_tx1)
    aa = np.mean(ret1, axis=0)
    for i in range(len(aa)):
        if np.isnan(aa[i]):
            aa[i] = aa[i-1]

    a = aa - np.mean(aa)
    phase = np.angle(a)
    phase_wrap = np.unwrap(phase)
    angle_calibrated, dynamic_vectors = calibration(a)

    pylab.figure()
    pylab.subplot(3, 3, 1)
    pylab.ylabel('Imag')
    pylab.xlabel('Real')
    pylab.plot(get_Real(a), get_Imag(a), 'b')
    pylab.xlim(-2, 2)
    pylab.ylim(-2, 2)

    pylab.subplot(3, 3, 2)
    pylab.title("angle")
    pylab.ylabel('angle/rad')
    pylab.xlabel('time')
    pylab.plot(phase, 'b')

    pylab.subplot(3, 3, 3)
    pylab.title("angle_wrap")
    pylab.ylabel('angle/rad')
    pylab.xlabel('time')
    pylab.plot(phase_wrap, 'b')

    pylab.subplot(3, 3, 4)
    pylab.title("distance")
    pylab.ylabel('dis/cm')
    pylab.xlabel('time')
    pylab.plot(phase_wrap * 5.64 / (4 * np.pi), 'b')

    pylab.subplot(3, 3, 5)
    pylab.ylabel('angle/rad')
    pylab.xlabel('calibration')
    pylab.plot(angle_calibrated, 'r')

    pylab.subplot(3, 3, 6)
    pylab.ylabel('dis/cm')
    pylab.xlabel('calibration')
    pylab.plot(np.unwrap(angle_calibrated), 'r')
    return np.unwrap(angle_calibrated) * 5.64 / (2 * np.pi)


if __name__ == '__main__':
    x = trace("../0928/rx1_3.dat")
    y = trace("../0928/rx2_3.dat")
