# coding:utf-8
from numpy import *
import boost

def loadSimpData():
    datMat = matrix([[1.0,2.1],[2.0,1.1],[1.3,1.0],[1.0,1.],[2.0,1.0]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat,labelMat

def adaClassify(dataToClass,classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = boost.splitDataSet(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst

        # print aggClassEst
    return sign(aggClassEst)

def plotROC(preStrengths,classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #保留的是绘制光标的位置
    ySum = 0.0 #用于计算AUC的值
    numPosClas = str(sum(array(classLabels)==1.0))

    yStep = 1.0/float(numPosClas) #y轴上的步长
    xStep = 1.0/float(len(classLabels)-int(numPosClas)) #x轴上的步长
    sortedIndicies = preStrengths.argsort() #获得排好序的索引

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep;delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--') # 定义x,y轴的范围
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep


dataMat,classLabels = loadDataSet('horseColicTraining2.txt')
classifierArr,aggClassEst = boost.adaBoostTrainDS(dataMat,classLabels,10)

plotROC(aggClassEst.T,classLabels) #画图

testArr,testLabels = loadDataSet('horseColicTest2.txt')
prediction10 = adaClassify(testArr,classifierArr)
# print prediction10.T
errArr = mat(ones((67,1)))
print "the error rate is: %.5f" %(errArr[prediction10 != mat(testLabels).T].sum()/shape(errArr)[0])
# result = adaClassify([0.0,0.0],classifierArr)
# print result
