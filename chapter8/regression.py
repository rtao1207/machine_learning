# coding:utf-8
from numpy import *

# 用来读取数据集，返回数据信息和类别信息
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))#获得每行的数据量，前面的数为x0,x1,最后一个为目标值y
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
    yMat = mat(yArr).T # 将y转换为矩阵并转置
    xTx = xMat.T*xMat # 计算xTx
    # 计算xTx的行列式，如果为0,那么为不可逆矩阵，计算逆矩阵时就会出错
    # linalg.det()来计算行列式
    if linalg.det(xTx) == 0.0:
        print"This matrix is singular, cannot do inverse"
        return
    else:
        ws = xTx.I * (xMat.T*yMat) # 计算w回归系数：w = (xT*x)的逆*(xT*y)
        # ws = linalg.solve(xTx,xMat.T*yMat) #计算回归系数的另一种实现

    return ws

# 局部加权线性回归函数
'''
    局部加权线性回归函数（Locally Weighted Linear Regression,LWLR）思想：给
    待测点附近的每个点赋予一定的权重，然后再在这个子集上基于最小均方差来进行普通的回归。
    LWLR使用“核”来对附近的点赋予更高的权重
'''
# 对单点进行估计
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr);yMat = mat(yArr).T
    m = shape(xMat)[0] # 获得训练集数据数量
    weights = mat(eye((m))) #创建权重的对角矩阵，初始化为1
    for j in range(m):
        diffMat = testPoint - xMat[j,:] #对每条待测数据，计算其与数据集中的每条数据计算距离
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) #计算每个样本点对应的权重值,大小以指数级衰减，k用来控制权重衰减的速度
    xTx = xMat.T*(weights*xMat) #计算xTx,加了权重
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular,cannot do inverse"
        return
    ws = xTx.I * (xMat.T*(weights*yMat)) #回归系数估计矩阵，加了权重
    return testPoint*ws #返回对待测数据的预测

# 对数据集中的所有点进行估计
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0] #获得测试集数据数量
    yHat = zeros(m) #初始化最后的预测结果矩阵为全0
    for i in range(m): #对每个待测样本
        yHat[i] = lwlr(testArr[i],xArr,yArr,k) #计算出预测值yHat
    return yHat #输出最终的结果

# 误差函数:用于计算预测误差的大小
def rssError(yActual,yEstimate):
    return ((yActual - yEstimate)**2).sum()

# 岭回归:用于对那些特征大于样本数的数据集，对xTx加入lam*I(单位矩阵)，计算回归系数
def ridgeRegres(xMat,yMat,lam = 0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0: #计算行列式是否为0.来决定是否为可逆矩阵
        print "This matrix is singular , cannot do inverse"
        return
    ws = denom.I * (xMat.T*yMat) #计算回归系数
    return ws

# 用于在一组lambda上进行测试
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    #对数据进行标准化处理：就是将每一个特征减去各自的均值并除以方差
    yMean = mean(yMat,0) #求y的均值
    yMat = yMat - yMean

    xMeans = mean(xMat,0) #求x的均值
    xVar = var(xMat,0) #求x的方差
    xMat = (xMat - xMeans)/xVar #对x进行标准化

    numTestPts = 30 #定义30个不用的lambda值：且以指数级变化
    wMat = zeros((numTestPts,shape(xMat)[1])) #初始化一个回归系数矩阵，保存所有的回归系数

    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T

    return wMat #返回回归系数矩阵

# 对数据集特征进行标准化的函数
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # 求均值
    inVar = var(inMat, 0)  # 求方差
    inMat = (inMat - inMeans) / inVar  # 进行了标准化
    return inMat

# 前向逐步线性回归
'''
    xArr:输入数据
    yArr:预测数据
    eps:表示每次迭代需要调整的步长
    numIt:迭代次数
'''
def stageWise(xArr,yArr,eps = 0.01,numIt = 100):
    xMat = mat(xArr); yMat = mat(yArr).T

    # 对y进行标准化处理
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    # 对x进行标准化处理，对特征进行均值为0，方差为1的标准化处理
    xMat = regularize(xMat) #调用标准化处理函数

    m,n = shape(xMat) # 获得行数和列数
    returnMat = zeros((numIt,n)) # 初始化要返回的权重矩阵

    ws = zeros((n,1)) # 同来保存w的值
    wsTest = ws.copy();wsMax = ws.copy() #建立了ws的两个副本，用于实现贪心算法

    #进行优化
    for i in range(numIt):
        print ws.T #每次迭代都打印w的值
        lowestError = inf #初始化最低误差为无穷大
        for j in range(n): #对每个特征
            for sign in [-1,1]: # -1或1中
                wsTest = ws.copy()
                wsTest[j] += eps*sign #更新权重：分别增加或减少该特征对误差的影响
                yTest = xMat*wsTest #计算预测值
                rssE = rssError(array(yMat),array(yTest)) #计算误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T

    return returnMat

# 标准回归函数的测试
def standRegresTest():
    xArr,yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr,yArr) # 根据最佳拟合直线来计算回归系数
    xMat = mat(xArr) #将x转换为矩阵
    yMat = mat(yArr) #将y转换为矩阵
    yHat = xMat*ws #获得y的预测值

    #利用numpy中的corrcoef(yEstimate,yActual)来计算预测值与真实值的相关性
    print corrcoef(yHat.T,yMat) #计算这两个序列的相关系数

    # 绘制数据集散点图以及最佳拟合直线
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],color='red')

    xCopy = xMat.copy()
    xCopy.sort(0) #按照x进行排序，默认为升序
    yHat2 = xCopy*ws
    plt.plot(xCopy[:,1],yHat2,color='b')
    plt.show()

# 局部加权线性回归函数测试
def locallyWeightedLinearRegressionTest():
    xArr,yArr = loadDataSet('ex0.txt')
    # print xArr
    # testRes = lwlr(xArr[0],xArr,yArr,1.0)
    # print testRes
    # testRes2 = lwlr(xArr[0],xArr,yArr,0.01)
    # print testRes2
    yHat = lwlrTest(xArr,xArr,yArr,0.01)
    # print yHat

    xMat = mat(xArr)
    # print xMat[:,1].argsort(0)
    sortedIndex = xMat[:,1].argsort(0) #argsort()函数是用来将数据进行从小到大的排序，然后取其索引
    # print xMat[sortedIndex]
    # print shape(xMat[sortedIndex][:,0,:])
    xSort = xMat[sortedIndex][:,0,:] # 三维转换成二维的
    # print xSort

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[sortedIndex],c='black') #这是预测的值，构造线
    ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='r') #这是真实的值，构造散点图
    plt.show()

# 预测鲍鱼的年龄
def fishPredict():
    abX,abY = loadDataSet('abalone.txt')
    yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    error01 =  rssError(array(abY[0:99]),yHat01.T)
    error1 =  rssError(array(abY[0:99]),yHat1.T)
    error10 =  rssError(array(abY[0:99]),yHat10.T)
    print error01
    print error1
    print error10
    '''
       使用较小的核产生了较小的误差，那么为啥不一直选择较小的核呢？
       因为较小的核容易造成过拟合，对新数据不一定能达到最好的预测效果；
       比如接下来使用新数据来看看效果如何
    '''
    new_yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    new_yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    new_yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    new_error01 = rssError(array(abY[100:199]), new_yHat01.T)
    new_error1 = rssError(array(abY[100:199]), new_yHat1.T)
    new_error10 = rssError(array(abY[100:199]), new_yHat10.T)
    print new_error01
    print new_error1
    print new_error10

    '''
        此时可以看到核大小等于10时的测试误差最小，但是它你在训练集上的误差确是最大的
    '''

# 岭回归的测试
def ridgeRegressionTest():
    abX, abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX,abY) #岭回归系数

    # print ridgeWeights
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights) #绘制不同的log(lambda)对应的回归系数分布
    plt.show()

# 前向逐步线性回归测试
def stageWiseTest():
    xArr,yArr = loadDataSet('abalone.txt')
    # returnMat = stageWise(xArr,yArr,0.01,200)
    returnMat = stageWise(xArr,yArr,0.005,1000)
    print returnMat

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)  # 绘制随着迭代次数变化回归系数变化
    plt.show()

if __name__ == '__main__':
    # standRegresTest()
    # locallyWeightedLinearRegressionTest()
    # fishPredict()
    # ridgeRegressionTest()
    stageWiseTest()
