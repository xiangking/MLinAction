#coding=utf-8

import numpy as np

#数据读取
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat



#该过程即为已经确定了一个alphas，在剩下的中选取另一个alphas构成SMO算法中选择两个来优化的过程
#：已经选择的alphas的下标；全部alphas的数码（要-1，毕竟下标从0算起）
#r:选择的下标
def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j



#剪辑aj，即使得得到的aj在[L,H]的范围内；具体方法参考统计学习方法127页7.108
#：需要剪辑的a；上限H；下限L
#r:剪辑后的aj
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj



#使用SMO算法进行SVM
#：数据列表；标签列表；权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）；容错率；最大迭代次数
#r:返回最后的b值和alpha向量
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()

    b = 0
    m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0

    while (iter < maxIter):
        alphaPairsChanged = 0

        for i in range(m):
            #计算预测值，利用的公式是alphas*y*(x*x)+b,可参考李航统计学方法中的7.26
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #计算出预测值与实际值之间的误差
            Ei = fXi - float(labelMat[i])

            #if内条件即为如果改变alphasi所产生的误差不大，那么该alphas就不是需要优化的alphas，选择其他alphas进行优化
            #如果误差超过容许范围，那么确认满足0<alphas<C的条件
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) \
                 or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #以下过程即挑选另一个alphasj并计算相应需要的参数
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])

                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #这里的理解可参考统计学习方法第126页，也就是求出alphas的上下界
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L==H: print "L==H"; continue  #如果上下界没有变化，就表示不需要更新
                # 根据公式计算未经剪辑的alphaj
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T \
                          - dataMatrix[i,:]*dataMatrix[i,:].T \
                          - dataMatrix[j,:]*dataMatrix[j,:].T

                if eta >= 0: print "eta>=0"; continue  #如果eta>=0,跳出本次循环

                #该公式可参考统计学习方法127页7.106
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                #剪辑alphas
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 如果改变后的alphaj值变化不大，跳出本次循环
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                # 否则，计算相应的alphai值
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                # 再分别计算两个alpha情况下对于的b值
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                # 如果0<alphai<C,那么b=b1
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                # 否则如果0<alphai<C,那么b=b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                # 否则，alphai，alphaj=0或C
                else: b = (b1 + b2)/2.0
                # 如果走到此步，表面改变了一对alpha值
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        # 最后判断是否有改变的alpha对，没有就进行下一次迭代
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter

    # 返回最后的b值和alpha向量
    return b,alphas



