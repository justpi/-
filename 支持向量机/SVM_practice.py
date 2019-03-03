# -*- coding: utf-8 -*-
'''
    作者:     李高俊
    版本:     1.0
    日期:     2018/12/5/
    项目名称： 支持向量机
'''


import numpy as np
import os
import random
import matplotlib.pyplot as plt
import math


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName,encoding="ISO-8859-1")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j


def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def SMOSimple(dataMat,labelMat,C,toler,maxIter):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat).transpose()
    b= 0
    m,n = dataMatrix.shape
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMatrix).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMatrix[i])
            if ((labelMat[i] * Ei) < -toler and (alphas[i] < C)) or ((labelMat[i] * Ei) > toler and
                                                                 (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMatrix).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMatrix[i] != labelMatrix[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i]-C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T -dataMatrix[i,:] * dataMatrix[i,:].T- \
                    dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMatrix[j] *(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMatrix[j] * labelMatrix[i] *(alphaJold -alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T \
                    - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej -labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T \
                    - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if (0 <alphas[i]) and (C > alphas[i]):  #i为支持向量
                    b = b1
                elif (0 <alphas[j]) and (C > alphas[j]):    #j为支持向量
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairChanged += 1
                print("iter:%d i:%d pairs changed:%d" %(iter,i,alphaPairChanged))
        if alphaPairChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" %iter)
    return b,alphas


def kernelTrans(X,A,kTup):  #高斯核函数
    m,n = X.shape
    K = np.mat(np.zeros((m,1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
            K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError ("Houston We Have a Problem -- That Kernel is not recongnized")
    return K



class optStruct:


    def __init__(self,dataMat,classLabels,C,toler,kTup):
        self.X = dataMat
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMat.shape[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)


    def calcEk(oS,k):
        fXk = float(np.multiply(oS.alphas,oS.labelMat).T * oS.K[:,k] + oS.b)
        Ek = fXk - float(oS.labelMat[k])
        return Ek


    def selectJ(i,oS,Ei):
        maxK = -1;maxDeltaE = 0;Ej = 0
        oS.eCache[i] = [1,Ei]
        validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = optStruct.calcEk(oS,k)
                deltaE = abs(Ei -Ek)
                if (deltaE > maxDeltaE):
                    maxK = k;maxDeltaE = deltaE;Ej = Ek
            return maxK,Ej
        else:
            j = selectJrand(i,oS.m)
            Ej = optStruct.calcEk(oS,j)
            return j,Ej


    def updateEk(oS,k):
        Ek = optStruct.calcEk(oS,k)
        oS.eCache[k] = [1,Ek]


def innerL(i,oS):
    Ei = optStruct.calcEk(oS,i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0)):
        j,Ej = optStruct.selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0,oS.alphas[j] + oS.alphas[i] -oS.C)
            H = min(oS.C,oS.alphas[i] + oS.alphas[j])
        if L == H:print("L == H");return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i]-\
            oS.K[j,j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] *(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        optStruct.updateEk(oS,j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
        oS.alphas[i] += oS.labelMat[j] *oS.labelMat[i] *(alphaJold -oS.alphas[j])
        optStruct.updateEk(oS,i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] -alphaIold) * oS.K[i,i]\
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] -alphaIold) * oS.K[i,j]\
             - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else:oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


def smoP(dataMat,labelMat,C,toler,maxIter,kTup = ("lin",0)):
    oS = optStruct(np.mat(dataMat),np.mat(labelMat).transpose(),C,toler,kTup)
    iter = 0
    entireSet = True
    alphaPairChange = 0
    while (iter < maxIter) and ((alphaPairChange > 0) or (entireSet)):
        alphaPairChange = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairChange += innerL(i,oS)
                print("fullSet,iter:%d i %d,pair changed %d" %(iter,i,alphaPairChange))
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairChange += innerL(i,oS)
                print("non-bound,iter:%d i %d,pair changed %d" % (iter, i, alphaPairChange))
            iter += 1
        if entireSet: entireSet = False
        elif (alphaPairChange == 0): entireSet = True
        print("iteration number: %d" %iter)
    return oS.b,oS.alphas


def calcW(alphas,dataArr,classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m,n = X.shape
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i] , X[i,:].T)
    return w


def calcy(w,b,dataArr):
    dataArr = np.mat(dataArr)
    predValue = dataArr * w + b
    if predValue > 0:
        return 1
    else:
        return -1


def testRbf(k1 = 1.3):
    dataArr,labelArr = loadDataSet("testSetRBF.txt")
    b,alphas = smoP(dataMat=dataArr,labelMat=labelArr,C=200,toler=0.0001,maxIter=1000,kTup = ('rbf',k1))
     dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSv = labelMat[svInd]
    print("there are %d Support Vectors" % sVs.shape[0])
    m,n = dataMat.shape
    errorCount = 0.0
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(labelSv,alphas[svInd]) +b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("训练集分类错误%d个" %errorCount)
    print("分类错误率为：%f" % (errorCount / m))
    dataArr,labelArr = loadDataSet("testSetRBF2.txt")
    errorCount = 0.0
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    m,n = dataMat.shape
    for i in range(m):
        kernelEval = kernelTrans(sVs,dataMat[i,:],("rbf",k1))
        predict = kernelEval.T * np.multiply(labelSv,alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    print("测试集上分类错误%d个" %errorCount)
    print("分类错误率为：%f" %(errorCount/m))


def main():
    testRbf(0.5)
    # dataSet,labelSet = loadDataSet("testSetRBF.txt")
    # # b,alphas = SMOSimple(dataSet,labelSet,0.6,0.001,100)
    # b,alphas = smoP(dataSet,labelSet,1,0.0001,40)
    # w = calcW(alphas,dataSet,labelSet)
    # print("预测值：",calcy(w,b,dataSet[6]))
    # print("实际值：",labelSet[6])
    # svmvector = []
    # svmlabel = []
    # postiveSet = []
    # neigetiveSet = []
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         svmvector.append(dataSet[i])
    #         svmlabel.append(labelSet[i])
    #     if labelSet[i] > 0:
    #         postiveSet.append(dataSet[i])
    #     else:
    #         neigetiveSet.append(dataSet[i])
    #
    # px = np.array(postiveSet)[:,0]
    # py = np.array(postiveSet)[:,1]
    # nx = np.array(neigetiveSet)[:,0]
    # ny = np.array(neigetiveSet)[:,1]
    # plt.scatter(px,py,marker='o')
    # plt.scatter(nx,ny,marker='s')
    # plt.scatter(np.array(svmvector)[:,0],np.array(svmvector)[:,1],marker='p')
    # plt.show()


if __name__ == '__main__':
    main()