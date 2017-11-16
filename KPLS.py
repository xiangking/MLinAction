#coding= utf-8

from numpy import*

# # #从txt文档中提取数据
# # def loadDataSet(fileName):      #general function to parse tab -delimited floats
# #     dataMat = []
# #     tempMat = []
# #     fr = open(fileName)
# #     for line in fr.readlines():
# #         print line
# #         for n in range(0,8):
# #             curLine = line.strip().split()
# #             tempMat.extend(curLine)
# #         # fltLine = map(float,curLine) #map all elements to float()
# #         dataMat.append(tempMat)
# #         tempMat = []
# #     return np.array(dataMat)
# def loadDataSet(fileName):      #general function to parse tab -delimited floats
#     dataMat = []
#     tempMat = np.zeros((2394,8))#assume last column is target value
#     index = 0
#     fr = open(fileName)
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         tempMat[index,:] = curLine[0:8]
#         index += 1
#         dataMat.append(tempMat)
#     return np.array(dataMat)
#
#
# n = loadDataSet('debutanizer_data.txt')
# print n[0]


def kernelTrans(X, kTup):
    m,n=shape(X)
    K=mat(zeros((m,m)))

    for i in range(m):
        Xi = X[i, :]
        for j in range(m):
            deltaRow = X[j,:] - Xi
            K[i,j] = deltaRow * deltaRow.T
            K[j,i] = K[i,j]
    K=exp(K/(-1 * kTup**2))

    return K

def kernelTrans_test(X_train, X_test, kTup):
    m,n=shape(X)
    t,g = shape(X_test)
    K=mat(zeros((t,m)))

    for i in range(t):
        Xi = X_test[i, :]
        for j in range(m):
            deltaRow = Xi - X[j,:]
            K[i,j] = deltaRow * deltaRow.T
    K=exp(K/(-1 * kTup**2))

    return K

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

dataSet = load_iris()
X, X_test, Y, Y_test = train_test_split(dataSet.data, dataSet.target, test_size=0.25, random_state=33)
X = matrix(X)

# # 测试PLS算法
# pls2 = PLSRegression(copy=True, max_iter=500, n_components=2, scale=False, tol=1e-06)
# pls2.fit(X, Y)
# U = pls2.y_scores_
# T = pls2.x_weights_
# Y_pred = pls2.predict(X)
# print T.shape



KX = kernelTrans(X, 1.3)
pls2 = PLSRegression(copy=True, max_iter=500, n_components = 2, scale=False, tol=1e-06)
pls2.fit(KX, Y)

# print pls2.coef_



U = pls2.y_scores_
T = pls2.x_weights_
print T.shape, U.shape, Y.shape
Y = matrix(Y)
Y_pred = KX*U*(T.T * KX * U).I *T.T * Y.T
# print Y_pred.shape
Y_pred2 = dot(T,T.T) * Y.T

Y_pred3 = pls2.predict(KX)

# print Y_pred2.shape
print Y_pred3.shape




err2 = mean_squared_error(Y_pred,Y.T)
print err2
Y_pred = pls2.predict(X_test)
print pls2.score(KX,Y.T)




X_test = matrix(X_test)
Y_test = matrix(Y_test)
K_nomral = kernelTrans(X_test, 1.39)
K_test = kernelTrans_test(X, X_test, 1.39)
print K_test.shape, K_nomral.shape

pred_test = pls2.predict(K_test)
print Y_test.shape

# pls3 = PLSRegression(copy=True, max_iter=500, n_components = 2, scale=False, tol=1e-06)
# pls3.fit(K_nomral, Y_test)
# U_test = pls3.y_scores_
# T_test = pls3.x_weights_

err3 = mean_squared_error(Y_test.T,pred_test)
print err3

# Y_pred_test = K_test*U*(T.T * KX * U).I *T.T * Y_test
Y_pred_test = K_test*U*(T.T * KX * U).I *T.T * Y.T

err4 = mean_squared_error(Y_pred_test,Y_test)
print err4













