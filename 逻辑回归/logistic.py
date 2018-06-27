#coding=utf-8
#logistic回归不要看带了回归两个字，但其实是分类算法，具体原理参考统计学方法

import numpy as np


#从txt中加载数据
def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

#定义sigmoid函数
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#logistic 的梯度上升算法
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)   #将输入数据转化为numpy的矩阵形式，方便后面计算
    labelMatrix = np.mat(classLabels).transpose()  #转置
    m,n = np.shape(dataMatrix)  #得到输入数据的行列
    alpha = 0.001  #学习率设置为0.001
    weight = np.ones((n,1))  #初始化权重
    maxCycles = 500  #最大计算轮数
    #梯度上升的整个算法，原理参考书上
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weight)  #得到计算的输出值
        error = labelMatrix - h  #得到实际值和计算值的差值
        weight = weight + alpha * dataMatrix.transpose()* error  #python中矩阵加上一个常数会自动将常数扩充到与矩阵相应的形式在相加

    return weight

#绘制决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    weights = weights.getA()  #将权重的数组的数据类型转化成ndarray类型
    dataMat,labelMat=loadDataSet()  #加载数据
    dataArr = np.array(dataMat)  #将数据转化成ndarray类型
    n = np.shape(dataArr)[0]  #得到数据个数（即行数）
    xcord1 = []; ycord1 = []  #先设置各类点的x,y存放数组，这里的x,y即是书上的x1,x2，只是将x2作为y
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:  #如果是一类，则将点的坐标放入xcord1中
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)  #将图分为一行一列
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s') #以xcord1和ycord2的数值作画
    ax.scatter(xcord2, ycord2, s=30, c='green')
    #下面三句代码是为了绘制直线
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')  #标注x,y轴的图注
    plt.ylabel('X2')
    plt.show()


#普通的随机梯度上升,大体步骤和原理与梯度上升相同，但是随机梯度上是随机选择一个来确定方向，而之前使用的批梯度下降是使用所有数据的梯度来得到前进的方向
#两者区别在于前者速度更快，但后者相对会更准确。所以在数据量大时，可以考虑使用随机梯度下降。比如深度学习中就会使用随机梯度下降。
def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


#改进的梯度上升算法，改进处已做注释
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #每次迭代都对学习率进行更改
            randIndex = int(np.random.uniform(0,len(dataIndex)))  #每次更新weight值时都随机选取数据和标签
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights



