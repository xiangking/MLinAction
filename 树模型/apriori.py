#coding= utf-8

from numpy import *

#加载数据集：与以往不同的是，它的每一个样本的长度是随机的
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


#将数据集中的项集的中元素挑出来组成集合（元素不重复）
#：数据集
#r ：被挑出来的元素组成的集合
def createC1(dataSet):
    C1 = []
    #遍历数据集，如果不在c1中，则添加到c1中
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    #对数组进行排序，从小到大排
    C1.sort()
    #冻结成一个数组输出，map的作用是将c1中的每一个元素转化为frozenset类型
    return map(frozenset, C1)


#计算支持度
#：全集；子集；最小支持度
#r :超过最小支持度的项集；项集的支持度集合字典
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid): #判断can是否是tid的子集，并使用has_key,如果can不在ssCnt中，就将其作为新统计量，并置1
                                                        #如果在，那就将其加1
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}
    #下面计算集合的支持度，并把超过最小支持度的存入supportData中
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


#得到所有有lk里的项集组成的集合：由于如果直接使用集合合并，会出现合并之后产生重复的集合
#如{0,1},{0,2},{1,2}如果使用一一取并集，则会产生3个{0,1,2}
#为了避免这种情况，通过比较每个项集的前k-2个元素，如果相同，则合并
#：频繁项集；k
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    # 第一次进入这个函数时，由于L中的项集都是单个元素，而且k-2为0，L1,L2都为空集一定相等，所以一定都是一一合并
    # 后面加入这个函数时，还是按前k-2项相等才合并
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()

            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

#通过支持度，选择频繁项集
#：数据集；最小支持度
#r ：频繁项集数组；所有项集的支持度字典
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2  #关于k为什么选择2还是有疑问，之后在看
    while (len(L[k-2]) > 0): #直到新加入的频繁项集是空集为止
        Ck = aprioriGen(L[k-2], k) #将频繁项集进行合并
        Lk, supK = scanD(D, Ck, minSupport) #再对通过合并频繁项集所得到的项集进行支持度的判断，看是否满足最小支持度的要求
        supportData.update(supK) #将新的支持度字典加入
        L.append(Lk) #将新的频繁项集加入
        k += 1
    return L, supportData





#关联规则生成函数
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    #存储所有关联规则
    bigRuleList = []

    for i in range(1, len(L)): #一条关联规则至少需要2个元素,故下标从1开始
        for freqSet in L[i]: #frequent为大小为i的项集
            H1 = [frozenset([item]) for item in freqSet] #对每个频繁项集构了,建只包含单个元素的集合,即可以出现在规则右边
            if (i > 1):
                #包含三个及以上元素的频繁集
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #包含两个元素的频繁集
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList



#处理包含两个元素的频繁集,计算规则的可信度，并过滤出满足最小可信度要求的规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []      #保存关联规则的列表

    for conseq in H:
        #对一个频繁集合,其分母都是一样的,supprot{1,2}/support{1},supprot{1,2}/support{2}
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # calc confidence

        if conf >= minConf:
            print freqSet - conseq, '-->', conseq, 'conf:', conf
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH



#根据当前候选规则集H生成下一层候选规则集,H是可以出现在规则右边的元素列表
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    #可以出现在规则右边的元素个数[1]=1,[1,2]=2,从1个元素增加直到小于freqSet的总个数都行
    m = len(H[0])

    if (len(freqSet) > (m + 1)):  # try further merging
        Hmp1 = aprioriGen(H, m + 1)  #产生大小为m+1的频繁集列表,(1,2->3)->(1->2,3),生成下一层H
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  #规则右边的元素个数还可以增加
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


# def pntRules(ruleList, itemMeaning):
#     for ruleTup in ruleList:
#         for item in ruleTup[0]:
#             print itemMeaning[item]
#         print "           -------->"
#         for item in ruleTup[1]:
#             print itemMeaning[item]
#         print "confidence: %f" % ruleTup[2]
#         print  # print a blank line


