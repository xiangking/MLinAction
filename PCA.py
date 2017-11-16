#coding= utf-8

import numpy as np

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return datArr


#利用协方差最大的方向进行降维
#：数据集；降维后的维度
#r ：利用降维后的矩阵反构出原数据矩阵；降维后的数据
def pca(dataMat,topNfeat=9999999):
    dataMat = np.mat(dataMat)

    meanVals = np.mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved,rowvar=0) #得到协方差矩阵

    eigVals,eigVects = np.linalg.eig(np.mat(covMat)) #得到基于协方差矩阵的特征值和特征向量（一列表示一个特征向量）
    eigValInd = np.argsort(eigVals) #对特征值进行排序（从小到大），返回其对应的数组索引（按从小到大顺序）
    eigValInd = eigValInd[:-(topNfeat + 1):-1] #从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的数组索引
    redEigVects = eigVects[:,eigValInd] #从特征向量数组中取出特征值较大的特征向量组成数组
    lowDDataMat = meanRemoved * redEigVects #将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
                                            #？为什么降维要用减去均值的数据集来降维
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return lowDDataMat,reconMat


import matplotlib
import matplotlib.pyplot as plt

def Pcaplot(dataMat,reconMat):
    dataMat = np.mat(dataMat)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #三角形表示原始数据点
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
    #圆形点表示第一主成分点，点颜色为红色
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0], marker='o',s=90,c='red')
    plt.show()






