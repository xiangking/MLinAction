#coding= utf-8

from numpy import *

def loadSimpData():
    datMat = mat([[ 1. ,  2.1],
                  [ 2. ,  1.1],
                  [ 1.3,  1. ],
                  [ 1. ,  1. ],
                  [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels




"""
#单层决策树分割数据的方法（不同于普通决策树的分类，单层决策树在分类过程中并没有复杂的选取最优特征进行分类）
#：dataMat：数据集；dimen：特征位置（也就是数据集的第几列）；threshVal：阈值；threshIneq：阈值选择模式
#r:预测值
"""
def stumpClassify(dataMat, feature, threshVal, threshIneq):
    m = shape(dataMat)[0]
    retArray = ones((m, 1))

    #阈值的模式，将小于某一阈值的特征归类为-1（单层决策树的目标变量的值为-1和0，
    #由于上面初始化时retArray的初值为1，所以这里只需要将一些位置赋值-1即可）
    if threshIneq=='lt':
        retArray[dataMat[:,feature]<=threshVal]=-1.0  #这里使用了布尔数组索引方法
    #将大于某一阈值的特征归类为-1
    else:
        retArray[dataMat[:,feature]>threshVal]=-1.0

    return retArray




"""
#构建单层决策树
#：dataMat：数据集；classLabels：标签值（目标变量值）；D：最初的权值
#r:最佳单层决策树相关信息的字典;最小错误率;决策树预测输出结果
"""
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)

    numSteps = 10.0                  #步长或区间总数
    bestStump = {}                   #最优决策树信息
    bestClasEst = mat(zeros((m,1)))  #最优单层决策树预测结果

    minError = inf   #将最小误差初始化为无穷大

    for i in range(n): #对特征进行循环
        #计算特征的最大最小值，并得到该特征的步长或者说区间间隔
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps

        for j in range(-1,int(numSteps)+1):#遍历各个步长区间；这里由于range（）会不包括右端的数，所以要加1
            #以下过程就是把所有的阈值都取一遍，看看哪个效果最好
            for inequal in ['lt', 'gt']: #两种阈值过滤模式
                threshVal = (rangeMin + float(j) * stepSize)  #阈值的求法；由于与阈值比较时使用大于，所以为了防止出现所有值都是同一个值的情况，所以使用-1
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)

                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 #这里使用布尔数组进行赋值，predictedVals == labelMat形成布尔数组，
                                                      # 所产生的效果就是两者相同的位置赋0值

                weightedError = D.T*errArr  #将权值矩阵乘以误差矩阵，就可以得到权值误差
                                            #（也就是筛选出有错误的位置的权值，把没有错误的位置的权值置0，然后加起来求和）

                if weightedError < minError:
                    minError = weightedError

                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i  #最好的特征列数
                    bestStump['thresh'] = threshVal  #最好的阀值
                    bestStump['ineq'] = inequal #最好的阈值的模式

    return bestStump,minError,bestClasEst  #返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果




#@dataArr：数据矩阵
#@classLabels:标签向量
#@numIt:迭代次数


"""
#算法主体（基本可参考李航p138到p139的公式即可）
#：dataArr：数据集；classLabels：标签值（目标变量值）；numIt：迭代次数
#r:弱分类器的组合列表
"""
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    #弱分类器相关信息列表
    weakClassArr=[]
    #获取数据集行数
    m=shape(dataArr)[0]
    #初始化权重向量的每一项值相等
    D=mat(ones((m,1))/m)
    #累计估计值向量
    aggClassEst=mat((m,1))
    #循环迭代次数
    for i in range(numIt):
        #根据当前数据集，标签及权重建立最佳单层决策树
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        # print("D:",D.T)
        #求单层决策树的系数alpha
        alpha=float(0.5*log((1.0-error)/(max(error,1e-16))))  #这里使用1e-16是为了防止分母为0的情况
        #存储决策树的系数alpha到字典
        bestStump['alpha']=alpha
        #将该决策树存入列表
        weakClassArr.append(bestStump)
        #打印决策树的预测结果
        # print("classEst:",classEst.T)

        #预测正确为exp(-alpha),预测错误为exp(alpha)，即增大分类错误样本的权重，减少分类正确的数据点权重
        #这里的公式可参考李航机器学习p139(8.4)；这里的expon就是为了使式子更有可读性
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        #更新权值向量
        D=multiply(D,exp(expon))
        D=D/D.sum() #除以求和是为了规范化

        #累加当前单层决策树的加权预测值
        aggClassEst=aggClassEst + alpha*classEst
        endClass = sign(aggClassEst)
        print "endClass",endClass.T

        #求出分类错的样本个数
        aggErrors=multiply(sign(aggClassEst)!= mat(classLabels).T,ones((m,1)))
        #计算错误率
        errorRate=aggErrors.sum()/m
        print "total error:",errorRate
        #错误率为0.0退出循环
        if errorRate==0.0:break

    #返回弱分类器的组合列表
    return weakClassArr


#以下是测试
if __name__ == '__main__':

    dataSet,label = loadSimpData()
    Arr = adaBoostTrainDS(dataSet,label)