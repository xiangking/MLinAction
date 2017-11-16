#coding= utf-8

import numpy as np


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#数据向量计算欧式距离
#：两个向量
#r：欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) # (vecA-vecB)^2的总和的开方


#随机初始化K个质心(质心满足数据边界之内)
#：数据集；所分的簇的个数
#r :质点向量数组
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]  #读取输入数据的列数，这里实际上是读取每一个样本点的维数（比如（x,
                              # y )形式的点，经过读取即变成2，得知它是在二维平面上的数据）
    centroids = np.mat(np.zeros((k,n))) #构建存放质点的向量

    for j in range(n): #遍历每一维，然后取得最小值，通过用最小值随机加上一个在数据范围内的值从而得到一个初始的质点
        minJ = min(np.array(dataSet[:,j]))
        rangeJ = float(max(dataSet[:,j]) - minJ) #得到该列数据的范围(最大值-最小值)
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1)) #k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值

    return centroids  #返回初始化得到的k个质心向量的数组


#Kmeans算法，通过计算每个样本到质点的距离，将它们进行分类，并计算每一类的均值重新作为质点，进行kmeans直到分簇结果不再变化
#：数据集；分簇个数；距离计算方法；初始化质点方法
#r ：质点；分簇结果
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    dataSet = np.array(dataSet)
    m,n = np.shape(dataSet) #得到数据集内包含的样本个数
    clusterAssment = np.mat(np.zeros((m,2)))

    centroids = createCent(dataSet, k) #得到质点向量的数组
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False

        for i in range(m): #遍历每一个样本
            minDist = np.inf
            minIndex = -1

            for j in range(k): #遍历每一个质点，得到质点和样本点之间的欧式距离平方，并得到最小的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            clusterAssment[i,:] = minIndex,minDist**2
        if clusterAssment[i, 0] != minIndex: clusterChanged = True #判断条件，如果有一个样本属于的簇和上次不同，就继续进行kmeans
        #上面两个for循环得到了每一个样本与质点之间最小的距离以及该质点的编号

        print centroids #输出质点


        for cent in range(k):#遍历k轮 ，得到新的质点
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]#将离那一些质点最近的样本挑出来
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #计算得到挑出来的样本数据的平均值，并利用它重新创造质点

    return centroids, clusterAssment


#二分聚类：先对整个数据集进行二分聚类，之后选择使得二分后聚类结果的误差最小的簇进行二分，直到簇的个数和要求的一样
#：数据集；聚类个数；所使用的距离函数
#r ：质点列表；簇数组
def biKmeans(dataSet, k, distMeas=distEclud):
    dataSet = np.mat(dataSet)
    m = np.shape(dataSet)[0] #得到数据集的样本数
    clusterAssment = np.mat(np.zeros((m,2))) #创建存放簇的矩阵
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] #获取数据集每一列数据的均值，组成一个长为列数的列表
    centList =[centroid0] #当前聚类列表为将数据集聚为一类


    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2 #计算每个样本与均值的欧式距离平方
                                                                           #并将其写入clusterAssment中

    #以下循环当第一次进入循环时，把整个数据集作为一个簇，对其进行二分聚类
    #当之后进入循环时，将选择聚类结果中误差最小的簇进行二分聚类
    while (len(centList) < k):
        # 将当前最小平方误差置为正无穷
        lowerSSE = np.inf
        # 遍历当前每个聚类
        for i in range(len(centList)):
            # 通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0], :]
            # 对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算该类划分后两个类的误差平方和
            sseSplit = sum(splitClustAss[:, 1])
            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit =  sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])

            # 划分第i类后总误差小于当前最小总误差
            if (sseSplit + sseNotSplit) < lowerSSE:
                # 第i类作为本次划分类
                bestCentToSplit = i
                # 第i类划分后得到的两个质心向量
                bestNewCents = centroidMat
                # 复制第i类中数据点的聚类结果即误差值
                bestClustAss = splitClustAss.copy()
                # 将划分第i类后的总误差作为当前最小误差
                lowerSSE = sseSplit + sseNotSplit

        # 从用来分割的旧簇中分出的两个新簇中的编号为1的作为新簇，编号为当前簇个数+1
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        # 从用来分割的旧簇中分出的两个新簇中的编号为0的取代旧簇的位置，依然使用旧簇的编号
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # 将 从用来分割的旧簇中分出的两个新簇中的编号为0的簇的质点加入质点数组，取代旧簇质点向量的位置
        centList[bestCentToSplit] = bestNewCents[0, :]
        # 将从用来分割的旧簇中分出的两个新簇中的编号为1的簇的质点向量添加数组中
        centList.append(bestNewCents[1, :])
        # 将新的两个簇的编号和误差加入到簇数组中
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0], :] = bestClustAss
        # 返回聚类结果

    return centList, clusterAssment
