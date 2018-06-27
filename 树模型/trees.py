# coding=utf-8


from math import log
import operator


# 建立测试用的集合
def createDataSet():
    dataSet = [[1, 1,1, 'yes'],
               [1, 1,0, 'yes'],
               [1, 0,1, 'no'],
               [0, 1,0, 'no'],
               [0, 1,0, 'no'],
               [0,1,1,'yes']]
    labels = ['no surfacing', 'flippers','wx']
    # change to discrete values
    return dataSet, labels




"""
#计算输入数据集的熵 即统计机器学习p60页——公式5.1
#：数据集（无label）
#r:经验熵
"""
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]    #这里指将dataSet列表里每一个小列表的最后一个值（也就是每一个数据最终是否是属于这个标签）
                                      # 赋值给currentLable
        #这里的if的作用是统计结果的数量（就本次编程来说，就是统计yes的个数和no的个数）
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #上面的程序通过空字典进行循环得到一个包含每个结果（如是和否）以及它们数量的字典
    #下面的循环正是利用上面得到的字典计算熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2

    return shannonEnt




"""
#划分数据集：
#： 代划分的数据集；划分数据集的特征；要求的值
#r :满足该特征等于value值的各样本的数组（注意：在这一过程中，由于已经用过该特征，所以将该特征有关信息删除，即去掉axis列）
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:     #这里是个对特征的分类分法，比如一个特征用1和0表示，
                                       #则通过该等式就可以将数据集分为该特征为1的一类和该特征为0的一类，
                                       #然后将这个特征去除后的列表返回
            reducedFeatVec = featVec[:axis]  #这几步的作用是删除该分类特征信息，因为已经使用过了，所以没用了，删掉
            reducedFeatVec.extend(featVec[axis + 1:])

            retDataSet.append(reducedFeatVec) #这部是将已经删除掉上面已经使用过的特征的数据重新做成一个数据集
                                              #最终循环完毕得到的是满足该特征等于value值且将该特征删去后的各样本的数组

    return retDataSet        #每一次输出即为一个叶节点下的一个类





"""
#用信息增益来判断特征，具体过程类似于统计学方法62的例5.2
#：数据集
#r ：最佳特征
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet) #先对整个数据集求它的经验熵
    bestInfoGain = 0.0 #刚开始的信息增益设为0
    bestFeature = -1

    #遍历所有的特征，每次针对该特征计算其信息增益
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#这里等于将dataSet中的一列提取出来
        uniqueVals = set(featList)  #得到由该特征结果组成的集合（如是不是光滑这个特征所组成的结果就是（0，1）
                                    #这里使用集合是为了去除相同的元素
        newEntropy = 0.0

        for value in uniqueVals: #计算每一种划分的条件信息熵
            subDataSet = splitDataSet(dataSet, i, value)#得到根据该特征和该类所得到的分割数量
            prob = len(subDataSet) / float(len(dataSet))#按统计学方法里的符号：Di/D
            newEntropy += prob * calcShannonEnt(subDataSet)#按统计学方法里的符号：H(D/A)
        infoGain = baseEntropy - newEntropy  #得到该类的信息熵增益

        if (infoGain > bestInfoGain):  # 这里是比较过程，从中选出最好的特征
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature # returns an integer




"""
# 当输入直接只有标签时调用此函数
#：标签值
#r:出现次数最多的标签值
"""
def majorityCnt(classList):
    classCount = {}

    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #将classCount按降序排列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #返回
    return sortedClassCount[0][0]





"""
#创造决策树
#  :数据集；特征标签
#r :决策树
"""
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #将分类的结果（例如本例中的yes和no）存储在这个数组中


    # 以下是判断树生成停止的条件：两个if
    if classList.count(classList[0]) == len(classList):

        return classList[0]  # count统计如果classList出现的次数，如果相等，那么也就是该分类标准下只有一个结果，
                             # 那就没必要进行下列步骤，直接返回即可

    if len(dataSet[0]) == 1:  #只剩下最终的标签时的情况
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet) #首先选取该轮的最优特征
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}} #以字典的键值对的形式创造决策树
    del (labels[bestFeat]) #删除标签中的对应节点名（因为在选取特征时也会删去该特征）

    featValues = [example[bestFeat] for example in dataSet] #将该特征的值一一提出
    uniqueVals = set(featValues) #使用集合去除那些相同的值

    for value in uniqueVals:
        subLabels = labels[:]  #为了防止在过程在修改label值，所以先copy一份
        #使用splitDataSet函数得到可以得到去除bestFeat后所得到的新数据集，然后再用来创建树
        newdata = splitDataSet(dataSet, bestFeat, value)
        myTree[bestFeatLabel][value] = createTree(newdata, subLabels)

    return myTree





"""
#用决策树判断类别
#：已经生成的决策树；特征标签（对应的节点名）；需要判别的数据
#r :类别
"""
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]   #将该节点名（也就是键值赋给firstStr)
    secondDict = inputTree[firstStr] #将节点下的只敢都赋给seconDict（也就是键值所带的值）
    featIndex = featLabels.index(firstStr) #返回这个键的在featLabels

    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    #判断valueOfFeat是不是字典，是字典表示还需要递归，非字典就表示已经到了最底的标签
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat

    return classLabel





"""
#保存生成的树模型
#：已经生成的决策树；保存的文件名
#r :无，在过程中保存
"""
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


"""
#使用保存的决策树模型
#：文件名
#r :读取的决策树模型
"""
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
