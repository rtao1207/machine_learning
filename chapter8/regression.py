# coding:utf-8
from numpy import *

# 用来读取数据集，返回数据信息和类别信息
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat,labelMat

# 用来计算最佳拟合直线
def standRegres(xArr,yArr):
    xMat = mat(xArr) # 将x转换为矩阵
    yMat = mat(yArr).T # 将y转换为矩阵
    xTx = xMat.T*xMat # 计算xTx
    # 计算xTx的行列式，如果为0,那么为不可逆矩阵，计算逆矩阵时就会出错
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    else:
        ws = xTx.I * (xMat.T*yMat) # 计算w回归系数

    return ws

xArr,yArr = loadDataSet('ex0.txt')
# print xArr
ws = standRegres(xArr,yArr)

xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat*ws #获得y的预测值
print corrcoef(yHat.T,yMat) #计算这两个序列的相关系数

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])

xCopy = xMat.copy()
xCopy.sort(0)
yHat2 = xCopy*ws
plt.plot(xCopy[:,1],yHat2)
plt.show()