# coding:utf-8

from numpy import *

# 通过阈值比较对数据进行分类，所有在阈值一边的数据会被分到类别-1，而在另一边的数据分到类别+1
def splitDataSet(dataMatrix,column,threshVal,operator):
    retArray = ones((shape(dataMatrix)[0],1)) # 初始化为值为1的数组
    if operator == "lt":
        retArray[dataMatrix[:,column] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,column] > threshVal] = -1.0
    return retArray

# 单层决策树生成函数：以最小误差作为衡量标准找到最优列、不等于符号、阈值和重估的分类标签
'''
     遍历splitDataSet()函数所有的可能输入值，并找到数据集上最佳的单层决策树
     这里的最佳是基于数据的权重向量D来定义的
     dataMatrix:数据集
     classLabels:分类标签
     D: 列向量每个元素的平均权重————1/总元素数
'''
def decisionTree(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix) # 数据集行数和列数
    numSteps = 10.0 #迭代步数
    bestFeat = {} #最优项列
    bestClass = mat(zeros((m,1))) #最优预测分类
    minError = inf # 初始化最小误差为+∞
    for i in range(n): # 对数据集中的每一个特征
        rangeMin = dataMatrix[:,i].min() #最小值
        rangeMax = dataMatrix[:,i].max() #最大值
        stepSize = (rangeMax - rangeMin)/numSteps #步长 =（最大值-最小值）/步长数
        for j in range(-1,int(numSteps)+1): # 对每个步长数迭代：-1到（numSteps）
            threshVal = (rangeMin + float(j) * stepSize) #计算阈值：（最小值+迭代步数*步长数）
            # operator操作符，有两个取值 gt大于, lt小于 ————分类分割符
            for operator in  ['lt','gt']:
                # 调用splitDataSet方法，小于，大于
                predictedVals = splitDataSet(dataMatrix,i,threshVal,operator)
                # 初始化误差集为一个全1向量
                errSet = mat(ones((m,1))) #错误列向量，用于保存预测结果与真实结果是否一致
                errSet[predictedVals == labelMat] = 0 # 如果predictedVals与真是的labelMat一致，则保存为0，否则为1
                weightedError = D.T*errSet # 将权重向量与错误向量对应的元素相乘并求和，就得到了数值weightedError
                # print "split: column %d,thresh %.2f, thresh operator: %s, the weighted error is %.3f" %(i,threshVal,operator,weightedError)

            # 将当前的错误率与已有的最小错误率进行比对，如果当前的值小，那么就在词典bestStump中保存该单层决策树
                if weightedError < minError:
                    minError = weightedError #更新最小误差为权重误差
                    bestClass = predictedVals.copy() #最优预测类
                    bestFeat['dim'] = i #最优列
                    bestFeat['thresh'] = threshVal #最优阈值
                    bestFeat['ineq'] = operator # 最优分隔符号（大于或者小于号）

    return bestFeat,minError,bestClass

# 通过修改D的值调整弱分类器的权重
# dataSet:数据集
# classLabels:分类标签
# numIt:迭代次数

def adaBoostTrainDS(dataSet,classLabels,numIt = 40):

    weakClassSet = [] # 初始化弱分类器
    m = shape(dataSet)[0]
    D = mat(ones((m,1))/m) #初始化D为平均权重
    aggClassSet = mat(zeros((m,1)))

    for i in xrange(numIt):

        bestFeat,error,classEst = decisionTree(dataSet,classLabels,D)
        # print "D:",D.T

        alpha = float(0.5*log((1.0-error)/max(error,1e-16))) #计算权重alpha，1e-16避免除0
        bestFeat['alpha'] = alpha

        weakClassSet.append(bestFeat)  #以数组形式存储弱分类器
        # print "classEst: ",classEst.T

        # 算法核心：D--权重修改公式：D*exp((+-)alpha)/sum(D)
        # +-号取决于是否错误，+表示正确划分，-错误划分
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #multiply矩阵点积
        D = multiply(D,exp(expon)) #下次迭代计算新的D
        D = D/D.sum()

        aggClassSet += alpha*classEst  #累积预测类
        # print "aggClassSet: ",aggClassSet.T

        # 如果x>0,sign(x) = 1,x<0,sign(x) = -1
        # 计算所有分类器的训练误差————累计误差
        aggErrors = multiply(sign(aggClassSet) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m  #计算总误差率
        print "total error: ",errorRate,"\n"

        if errorRate == 0.0: #如果是0，则划分完毕，跳出循环
            break

    return weakClassSet,aggClassSet