#coding= utf-8

# 比较重要，所以记一下笔记：这个启发式算法的核心在于对于整个数据集遍历比较容易，
# 而对于那些处于间隔边界上的点，我们还需要事先将这些点对应的alpha值找出来，
# 存放在一个列表中，然后对列表进行遍历；
# 此外，在选择第一个alpha值后，算法会通过一个内循环来选择第二个值，
# 在优化的过程中依据alpha的更新公式αnew,unc=aold+label*(Ei-Ej)/η，
# (η=dataMat[i,:]*dataMat[i,:].T+dataMat[j,:]*dataMat[j,:].T-2*dataMat[i,:]*dataMat[j,:].T),
# 可知alpha值的变化程度更Ei-Ej的差值成正比，
# 所以，为了使alpha有足够大的变化，选择使Ei-Ej最大的alpha值作为另外一个alpha。
# 所以，我们还可以建立一个全局的缓存用于保存误差值，便于我们选择合适的alpha值

import numpy as np
from time import sleep


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



#创建一个类来保存重要信息
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters

        self.X = dataMatIn #输入数据
        self.labelMat = classLabels  #label值
        self.C = C  #惩罚参数
        self.tol = toler  #容错的范围
        self.m = np.shape(dataMatIn)[0]  #数据个数
        self.alphas = np.mat(np.zeros((self.m, 1)))  #创建存放alphas的矩阵
        self.b = 0  #就是b
        self.eCache = np.mat(np.zeros((self.m, 2)))  # first column is valid flag


#格式化计算误差的函数，方便多次调用
#：对象；第k个数据
#r:误差Ek
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * \
                (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]  #提取eCache的第一列

    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue   # 这里很直观，如果K等于i,说明是同一个，没必要计算
            Ek = calcEk(oS, k)    #计算误差
            deltaE = abs(Ei - Ek) #得到Ek与Ei差的绝对值
            #选取与Ei差距最大的样本
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek

        return maxK, Ej

    else:  #否则，就从样本集中随机选取alphaj
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)

    return j, Ej

def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]



#内循环寻找alphaj
def innerL(i,oS):
    #计算误差
    Ei=calcEk(oS,i)
    #违背kkt条件,选择的第一个i需要违背KKT条件，以使得我们更好地更新
    if(((oS.labelMat[i]*Ei<-oS.tol)and(oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol)and(oS.alphas[i]>0))):
        j,Ej=selectJ(i,oS,Ei)

        alphaIold=oS.alphas[i].copy()
        alphaJold=oS.alphas[j].copy()

        #计算上下界
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])

        if L==H:print("L==H");return 0 #如果上下界相等，说明没有更新的必要
        #计算两个alpha值
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T

        if eta>=0:print("eta>=0");return 0
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)

        updateEk(oS,j)

        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print("j not moving enough");return 0

        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]* (alphaJold-oS.alphas[j])

        updateEk(oS,i)
        #在这两个alpha值情况下，计算对应的b值
        #注，非线性可分情况，将所有内积项替换为核函数K[i,j]
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
                    oS.X[i,:]*oS.X[i,:].T-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.X[i,:]*oS.X[j,:].T

        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
                    oS.X[i,:]*oS.X[j,:].T-\
                    oS.labelMat[j]*(oS.alphas[j]-alphaJold)*\
                    oS.X[j,:]*oS.X[j,:].T

        if(0<oS.alphas[i])and (oS.C>oS.alphas[i]):oS.b=b1
        elif(0<oS.alphas[j])and (oS.C>oS.alphas[j]):oS.b=b2
        else:oS.b=(b1+b2)/2.0

        #如果有alpha对更新
        return 1
             #否则返回0
    else: return 0


#SMO外循环代码
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    #保存关键数据
    oS=optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler)
    iter=0
    entireSet=True
    alphaPairsChanged=0

    #选取第一个变量alpha的三种情况，从间隔边界上选取或者整个数据集
    while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged=0 #没有alpha更新对

        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged+=innerL(i,oS)
            print "fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)

        else:
            #统计alphas向量中满足0<alpha<C的alpha列表
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged+=innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter+=1
        if entireSet:entireSet=False
        #如果本次循环没有改变的alpha对，将entireSet置为true，
        #下个循环仍遍历数据集
        elif (alphaPairsChanged==0):entireSet=True
        print "iteration number: %d" % iter

    return oS.b,oS.alphas


