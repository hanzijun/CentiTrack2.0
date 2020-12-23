#coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition.pca import svd_flip

def mean_data(data):
    return np.mean(data,axis=0)

"""
參数：
    - XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征
    - k：表示取前k个特征值相应的特征向量
返回值：
    - finalData：參数一指的是返回的低维矩阵，相应于输入參数二
    - reconData：參数二相应的是移动坐标轴后的矩阵
"""

def pca(XMat, k):
    average = mean_data(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)   #计算协方差矩阵
    featValue, featVec = np.linalg.eigh(covX)  #求解协方差矩阵的特征值和特征向量
    index = np.argsort(-featValue) #依照featValue进行从大到小排序
    if k > n:
        print ("k must lower than feature number")
        return
    else:
        #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
        selectVec = np.array(featVec.T[index[:k]]) #所以这里须要进行转置
        finalData = np.dot(XMat, selectVec.T)
        # reconData = (finalData * selectVec) + average
    return finalData

def PCAtest(XMat, k):
    average = mean_data(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    cov = np.dot(data_adjust.T,data_adjust)/(m-1)
    u, d, v = np.linalg.svd(cov, full_matrices=False)
    u, v = svd_flip(u, v)
    index = np.argsort(-d)
    final = np.dot(XMat, u[:, index[:k]])
    return final





