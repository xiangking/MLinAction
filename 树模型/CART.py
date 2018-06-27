#coding= utf-8

import numpy as np

#从txt文档中提取数据
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return np.array(dataMat)



#********以下是回归树代码**************************************************************／

"""
#叶结点生成函数
#p:dataSet数据集
#r:返回目标标签均值作为叶结点
"""
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])



"""
#计算数据集目标标签的总方差（下面直接写总方差，不在赘述是数据集目标标签的总方差了）
#p:数据集
#r:总方差值（将只有一个的方差值扩充到与样本个数相同的个数）
"""
def regErr(dataSet):
    #先用np.var得出均方差，然后乘以样本数就等于总方差了
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]



"""
#回归树预测
#p:dataSet：数据集
#r:叶结点的值
"""
def regTreeEval(treeModel, inDat):
    return float(treeModel)


#********以下是模型树代码**************************************************************／

"""
#模型树叶节点所用的函数
#p:dataSet：数据集
#r:回归系数;数据集矩阵;目标变量值矩阵
"""
def linearSolve(dataSet):
    # 获取数据行与列数
    m, n = np.shape(dataSet)
    # 构建大小为(m,n)和(m,1)的矩阵
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))

    # 数据集矩阵的第一列初始化为1，偏置项；每个样本目标变量值存入Y
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1].reshape(m,1) #多维数组当按行，或列取取出来是一维时，都会变成普通的一维数组，为了后面的计算，必须纠正其维度

    # 对数据集矩阵求内积
    xTx = X.T * X
    # 计算行列式值是否为0，即判断是否可逆
    if np.linalg.det(xTx) == 0.0:
        # 不可逆，打印信息
        print('This matrix is singular,cannot do inverse,\n\
                try increasing the second value if ops')


    # 可逆，计算回归系数
    ws = (xTx).I * (X.T * Y)

    # 返回回归系数;数据集矩阵;目标变量值矩阵
    return ws, X, Y


"""
#模型树的叶结点模型
#p:dataSet：数据集
#r:回归系数
"""
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    #返回该叶结点线性方程的回归系数
    return ws




"""
#模型树的误差计算函数
#p:dataSet：数据集
#r:返回误差的平方和，平方损失
"""
def modelErr(dataSet):
    # 构建模型树叶节点的线性方程，返回参数
    ws, X, Y = linearSolve(dataSet)
    # 利用线性方程对数据集进行预测
    yHat = X * ws
    # 返回误差的平方和，平方损失
    return sum(np.power(Y - yHat, 2))




"""
#模型树预测
#p:treeModel：所使用的树模型; inDat:数据集
#r:叶结点的值
"""
def modelTreeEval(treeModel, inDat):
    #获取输入数据的列数
    n = np.shape(inDat)[1]
    #构建n+1维的单列矩阵，因为要将第一列设置为1（值得注意的是这里的n的大小和上面创建时n的大小不用，因为这里的数据集最后一行不是目标变量）
    X = np.mat(np.ones((1, n + 1)))
    #第一列设置为1
    X[:, 1:n + 1] = inDat
    # 返回浮点型的回归系数向量
    return float(X * treeModel)

# ********以下是创建树的代码**************************************************************／

"""
#取出该数据集中特征比value小的样本
#p:数据集，特征，用来划分的值
#r:两个存放的数组
"""
def binSplitDataSet(dataSet, feature, value):
    #先用dataSet[:,feature] > value得到特征大于value的布尔矩阵
    #再用np.nonzero（）方法得到位置，最后提取该行存放再mat0中
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1



"""
#选择最佳切分特征和最佳特征取值
#p:dataSet：数据集；leafType：生成叶结点的类型；
   errType：计算误差的类型，默认为总方差类型；ops：用户指定的参数，默认tolS=1.0，tolN=4
   （这里的ops被称为预剪裁，为的就是防止树过拟合）
#r:最佳切分特征及最佳切分特征取值
"""
def chooseBestSplit(dataSet, leafType, errType, ops=(1,4)):
    tolS = ops[0]  #容许的误差下降值
    tolN = ops[1]  #最少切分样本数4

    if len(set(dataSet[:,-1].T.tolist())) == 1: #判断该数据集结果目标变量是否就只有一个（比如是和否，如果只有否或者是那就不需要建树了）
        return None, leafType(dataSet)

    m,n = np.shape(dataSet)  #得到数据集的样本个数和特征数（包含结果标签）
    S = errType(dataSet)     #得到总方差
    bestS = np.inf
    bestIndex = 0
    bestValue = 0

    #这里不删除特征，也就是说特征可以循环再分
    for featIndex in range(0, n-1):
        #按该特征的取值进行循环（例如0和1的话，就是分别取0和1，然后将数据集进行分类）
        for splitVal in set(dataSet[:,featIndex]):
            #以该特征，特征值作为参数对数据集进行切分为左右子集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            #求出总方差，然后做对比，如果比之前求出来的方差要小，那么就选取这个splitVal作为分割值
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    #如果切分后比切分前误差下降值未达到tolS
    if (S - bestS) < tolS:
        #不需切分，直接返回目标标签均值作为叶结点
        return None, leafType(dataSet) #exit cond 2

    #检查最佳特征及特征值是否满足不切分条件，这里我感觉貌似没什么用，因为上面已经检验过了，暂时不思考这个问题
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)

    return bestIndex,bestValue



"""
#创建树（先选取此时用以划分的最佳特征；然后是选取用来划分的值（比如这个特征>划分值时，归到右边的树；<=时，归到左边））
#p:数据集，特征，用来划分的值
#r:两个存放的数组
"""
def createTree(dataSet, leafType, errType, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  #选择最佳切分特征和最佳特征取值，这里将特征选为叶节点

    if feat == None: return val  #如果函数判断不需要切分，那么就直接返回目标变量的均值
    #下面是创建树的过程
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #根据最佳特征和特征值将数据集切分
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree


#********以下是用树进行预测的代码**************************************************************／

"""
#用来预测
#p:tree：树回归模型；inData：输入数据；modelEval：叶结点生成类型
#r:返回误差的平方和，平方损失
"""
def treeForecast(tree, inData, modelEval):
    #如果当前树为叶结点，生成叶结点
    if not isTree(tree): return modelEval(tree, inData)

    # 非叶结点，对该子树对应的切分点对输入数据进行切分
    if inData[tree['spInd']] > tree['spVal']:
            # 该树的左分支为非叶结点类型
        if isTree(tree['left']):
                # 递归调用treeForeCast函数继续树预测过程，直至找到叶结点
            return treeForecast(tree['left'], inData, modelEval)
            # 左分支为叶结点，生成叶结点
        else:
            return modelEval(tree['left'], inData)
        # 小于切分点值的右分支
    else:
         # 非叶结点类型
        if isTree(tree['right']):
                # 继续递归treeForeCast函数寻找叶结点
            return treeForecast(tree['right'], inData, modelEval)
            # 叶节点，生成叶结点类型
        else:
            return modelEval(tree['right'], inData)



"""
#使用树进行预测
#p:trees：树模型；testData：测试数据集；modelEval：模型树的预测函数
#r:预测值
"""
def createForecast(tree,testData,modelEval):
    #测试集样本数
    m=len(testData)
    #初始化行向量各维度值为1
    yHat=np.mat(np.zeros((m,1)))
    #遍历每个样本
    for i in range(m):
        #利用树预测函数对测试集进行树构建过程，并计算模型预测值
        yHat[i,0]=treeForecast(tree,np.mat(testData[i]),modelEval)
    #返回预测值
    return yHat


#********以下是后剪裁的函数**************************************************************／

"""
#判断是该对象的数据类型是否为字典
#p:obj：该对象
#r:布尔值（关于时候是字典类型）
"""
def isTree(obj):
    return (type(obj).__name__ == 'dict')



"""
#获得左右边的平均值
#p:tree：树模型
#r:平均值
"""
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])   #如果右边是树类型，那么继续迭代
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])      #如果左边是树类型，那么继续迭代

    return (tree['left'] + tree['right']) / 2.0   #将左边的值和右边的值求平均




"""
#获得左右边的平均值
#p:tree：树模型
#r:平均值
"""
def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree)  #如果树

    #左右分支中有非叶子结点类型(也就是字典类型），就使用binSplitDataSet分割数据集（这里其实就是使用树来看testData最后的目标变量的值）
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    #并且使用迭代，继续生成树
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)

    #如果已经是子叶结点类型，那么
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        #这里就是计算损失，用的是计算实际值和预测值的平方差（这里计算的是被放到左边的所有样本的损失和右边的所有样本的损失）
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'], 2))

        #计算预测的目标变量的目标值的平均数
        treeMean = (tree['left'] + tree['right']) / 2.0
        #计算被放到这一节点的所有样本的实际值与这个平均值预测值直接的损失
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        #如果分割后的损失没有不分割的小，那么就选择裁剪，选择不按照该节点分割，而是直接将节点变为结点
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree


#以下是测试
if __name__ == '__main__':

 # 创建回归树测试
 dataSet = loadDataSet('CARTdataSet2.txt')
 myTree = createTree(dataSet,regLeaf,regErr)
 print myTree

 # 树剪裁测试
 dataSet = loadDataSet('CARTdataSet2test.txt')
 dataSet = np.mat(dataSet)
 print dataSet.shape
 # myTree = prune(myTree,dataSet)
 # print myTree

 # #创建模型树
 # dataSet = loadDataSet('CRATbikeSpeedVsIq_train.txt')
 # myTree = createTree(dataSet,modelLeaf,modelErr,(1,20))
 # print "myTree = ",myTree
 # testSet = loadDataSet('CRATbikeSpeedVsIq_test.txt')
 # yHat = createForecast(myTree,testSet[:,0],modelTreeEval)
 # print "y = ",yHat
 # print np.corrcoef(yHat,testSet[:,1],rowvar=False)[0,1]



