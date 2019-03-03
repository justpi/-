# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018///
    项目名称： 患疝病的马的死亡率预测
'''
import logRegres
import numpy as np
import matplotlib.pyplot as plt


def classifyVector(inX,weight):
    prob = logRegres.sigmoid(sum(inX * weight))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest(numIter):
    frTrain = open('horseColicTraining.txt',encoding='ISO-8859-1')
    frTest = open('horseColicTest.txt',encoding='ISO-8859-1')
    trainingMat = []
    trainingLabel = []
    for line in frTrain.readlines():
        lineArr = line.strip().split('\t')
        lineArr = [float(i) for i in lineArr]
        trainingMat.append(lineArr[:-1])
        trainingLabel.append(int(lineArr[-1]))
    trainWeight = logRegres.stocGradAscent1(dataMat=np.array(trainingMat),labelMat=trainingLabel,numIter=numIter)
    errorCount = 0.0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        lineArr = line.strip().split('\t')
        lineArr = [float(i) for i in lineArr]
        if int(classifyVector(np.array(lineArr[:-1]),weight=trainWeight)) != int(lineArr[-1]):
            errorCount += 1
    errorRate = (errorCount)/(numTestVec)
    return errorRate

def main():
    errorRate = []
    for i in range(500):
        errorRate.append(colicTest(i))
    plt.plot(list(range(500)),errorRate)
    plt.show()

if __name__ == '__main__':
    main()