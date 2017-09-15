# coding:utf-8
import math
from numpy import *

# loadDataSet()函数主要用于按行读取testSet.txt文件，
# 每行的前两个值分别是X1，X2，第三个值是数据对应的类别标签
def loadDataSet():
    dataMat = [];labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) # 将X0设置为1.0
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# sigmoid()激活函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''
    梯度上升算法是对整个样本计算梯度，迭代n次，以更新回归系数：
      算法实现：每个回归系数初始化为1；
               重复R次：
                    计算整个数据集的梯度；
                    使用alpha*gradient更新回归系数的向量
               返回回归系数

    而随机梯度上升算法是一次仅用一个样本计算梯度，以更新回归系数：
      算法实现：所有回归系数初始化为1:；
               对数据集每个样本：
                    计算该样本的梯度；
                    使用alpha*gradient更新回归系数值
'''
# 梯度上升算法
# dataMat:表示一个2维100×3的Numpy数组，每列分别代表每个不同的特征（X0，X1，X2），每行代表每个训练样本
# labelMat:表示一个2维的1×100的行向量，每列代表每个样本所属的类别
def gradAscent(dataMat,labelMat):
    dataMatrix = mat(dataMat) # 转换为矩阵
    labelMat = mat(labelMat).transpose() # 行向量转换为列向量矩阵
    m,n = shape(dataMatrix) # 获得矩阵的行数和列数

    alpha = 0.001 #步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1)) # 初始化权值为3*1的列向量

    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) #h为一个列向量
        error = (labelMat - h)
        weights = weights + alpha*dataMatrix.transpose() * error # 对权重进行更新
    return array(weights) #因为weights是矩阵，将其转为数组

# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix) #获得数组的行数和列数
    alpha = 0.01 #步长
    weights = ones(n) #初始化为单位矩阵
    for i in range(m): # 对每个样本
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
        # print "epoch:",i," error:",error
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i)+0.01 #alpha在每次迭代时都会调整，会随着迭代次数不断的减小，但是不会减为0
            randIndex = int(random.uniform(0,len(dataIndex))) # 随机选取样本来更新回归系数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])

    return weights

# 画出最佳拟合的直线及其散点图
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr = mat(dataMat)

    m,n = shape(dataMat) # 获得矩阵的行数和列数
    xcord1 = [];ycord1 = [] # 保存类别为1的样本信息
    xcord2 = [];ycord2 = [] # 保存类别为0的样本信息
    for i in range(m): # 对每个样本
        if int(labelMat[i]) == 1: # 如果其类别为1
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2]) #
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111) #子图，对整个图进行分割布局
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s') #制作类别为1的数据散点图
    ax.scatter(xcord2,ycord2,s=30,c='green')          #制作类别为0的数据散点图
    x = arange(-3.0,3.0,0.1)  #x轴的取值范围，每隔1
    y = (-weights[0] - weights[1]*x)/weights[2] # 最佳拟合直线：0 = w0*x0+w1*x1+w2*x2,已知x0=1,x1,求x2即可

    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:return 1.0
    else:return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,numIter=500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1

    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f"%errorRate
    return errorRate

def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is:" \
          "%f" % (numTests,errorSum/float(numTests))

data,label = loadDataSet()
# print data
# print label


# weights = gradAscent(data,label)
# weights2 = stocGradAscent0(array(data),label)
weights3 = stocGradAscent1(array(data),label,numIter=500)
#
plotBestFit(weights3)
# plotBestFit(weights2)

multiTest()