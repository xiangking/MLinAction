#coding=utf-8
from numpy import *

#读取数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#标准回归
def standRegres(xArr,yArr):
    xMat = mat(xArr)  #将输入的数组转化为mat矩阵形式
    yMat = mat(yArr).T  #.T是求转置
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  # linalg.det(xTx)返回的是矩阵xTx的行列式的值
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

#画图1对于普通回归
import matplotlib.pyplot as plt
def plotRegres(xMat,yMat,ws):
    xMat = mat(xMat)
    yMat = mat(yMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0]) #.flatten().A[0]会将矩阵变为一行的数组，然后将其变为从二维数组变成一维
#以下是绘制回归线
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()


#画图2对于局部加权回归,绘制回归线时直接使用y而不是通过计算
def plotRegres2(xMat,yMat,y):
    xMat = mat(xMat)
    yMat = mat(yMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0]) #.flatten().A[0]会将矩阵变为一行的数组，然后将其变为从二维数组变成一维
#以下是绘制回归线
    strInd = xMat[:,1].argsort(0)
    xSort = xMat[strInd][:,0,:]
    ax.plot(xSort[:,1],y[strInd])
    plt.show()

#局部加权回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    m = shape(xMat)[0]
    weights = mat(eye((m)))  #生成对角矩阵
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #计算选择的这一点与数据集所有点之间的差值
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) #得到权重
    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return

    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

#该方法的作用是将局部加权中的y放到一个数组中
def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


#岭回归,为解决特征比样本点多时会出现无法求逆的问题，通过加入正则项以确保可以进行求逆运算
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    #利用值-平均值然后再除以方差使得每一特征的重要性得到均衡
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    #以下就是取不同的lamd得到不同的权重值
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat