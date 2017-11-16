#coding=utf-8



import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")





"""
#得到叶结点数（实际就是得到最终的结果数，比如本例中最后有多少个yes和n）
#p：已经生成的决策树
#r：叶结点数
"""
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        if type(secondDict[ key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1

    return numLeafs




"""
#得到树的深度
#p：已经生成的决策树
#r：树的深度
"""
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1

        if thisDepth > maxDepth: maxDepth = thisDepth

    return maxDepth








"""
#画箭头
#p:文本（这里就是节点名）；文本所在坐标；箭头所在坐标（注意，由于这里改变了箭头线的样式，所以此时指的是末尾坐标）；文本框样式
#r:无；在过程中生成注释信息
"""
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)





"""
#画叶节点之间的信息（在本例中就是指0和1）
#p:文本所在坐标；箭头所在坐标（注意，由于这里改变了箭头线的样式，所以此时指的是末尾坐标）；文本
#r:无；在过程中生成文本信息
"""
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)





"""
#绘制决策树图
#p:生成的树；箭头所在坐标（注意，由于这里改变了箭头线的样式，所以此时指的是末尾坐标）；文本信息
#r:无；在过程中生成决策树图
"""
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  #得到叶结点树
    depth = getTreeDepth(myTree)    #得到树的深度
    firstStr = myTree.keys()[0]     #得到树的节点名
    #计算文本所在坐标,也就是叶节点所在位置（这里的计算方式其实就是取在该叶节点下结果们最左边的位置+最右边的位置，然后除以2）
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)


    #以下两条就是绘制叶节点的过程
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    plotMidText(cntrPt, parentPt, nodeTxt)


    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  #以1／D的速度更新y

    for key in secondDict.keys():
        #以下是遍历的树，如果还未到底，就迭代；已经到底，就绘制结果
        if type(secondDict[key]).__name__ == 'dict':     #判断其类型是否为字典（也就是是否到底）
            plotTree(secondDict[key], cntrPt, str(key))  #如果是字典（也就意味着还没到底，那就继续迭代）

        else:  #以下是更新结果（比如在本例中就是绘制no和yes的图），以1／W的速度更新
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            #以下用来绘制结果
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD  #当一个分支更新完成后，要让y回到之前的状态，好用来另一分支





"""
#绘制决策树主要流程代码
#p:生成的树
#r:无；在过程中生成决策树图
"""
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  #用白色背景生成图
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False)  #frameon有边框还是无边框，**axprops在这里就是不让坐标轴出现的作用

    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0

    plotTree(inTree, (0.5, 1.0), '')  #调用plotTree创建树图形

    plt.show()


# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0:{'m':{0:{'m':{0:'no',1:'yes'}},1:'yes'}} , 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]

    # createPlot(thisTree)