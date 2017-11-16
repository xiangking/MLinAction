#coding=utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group = np.array([[1.0,1.1],
                      [1.0,1.0],
                      [0,0],
                      [0,0.1]])
    label = ['A','A','B','B']

    return group, label


#创建knn分类方法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet #tile是为了初始值变化到与dataSet同一矩阵形式；
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5#到此为止求出了各点到初始点的欧式距离

    sortedDistIndicies = distances.argsort() #对上面得到的各点距离按从小到大排列,返回各数值对应的大小序号
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)

    return  sortedClassCount[0][0]


#读取数据
def file2matrix(filename):
    fr = open(filename) #打开文件
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 得到数据的行数
    returnMat = np.zeros((numberOfLines, 3))  # 建立与输入对应的0矩阵
    classLabelVector = []  # 建立存放分类特征的数组
    index = 0

    for line in arrayOLines:
        line = line.strip()  # 去除回车
        listFromLine = line.split('\t')  # 以\t划分数据，每次读取以\t为终止
        returnMat[index, :] = listFromLine[0:3]  # 每次循环将每行除分类特征外的数据写入returnMat中
        classLabelVector.append(int(listFromLine[-1])) # 每次将一行中的分类特征加入classLabelVector中
        index += 1

    return returnMat, classLabelVector


#归一化 归一化结果=（数值-最小值）／最大值-最小值
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 按列取最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals # 取得范围
    normDataSet = np.zeros((np.shape(dataSet))) # 建立后续存放用的0矩阵
    m = dataSet.shape[0] # 行数
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))

    return normDataSet


#测试和评分
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount










