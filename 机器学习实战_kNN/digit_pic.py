# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018/11/20/
    项目名称： 手写识别系统
'''
import numpy as np
import knn_practice as knn
import pandas as pd
import os

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i + j] = int(lineStr[j])
    return returnVect


def handwriteClassifier():
    hwLabels = []
    trainingFilelist = os.listdir('./trainingDigits/')
    m = len(trainingFilelist)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFilelist[i]
        fileStr = fileNameStr.split('.')[0]
        numStr = int(fileStr.split('_')[0])
        hwLabels.append(numStr)
        trainingMat[i,:] = img2vector('./trainingDigits/%s' %fileNameStr)
    testFilelist = os.listdir('./testDigits/')
    mTest = len(testFilelist)
    errorCount = 0.0
    for  i in range(mTest):
        fileNameStr = testFilelist[i]
        fileStr = fileNameStr.split('.')[0]
        numStr = int(fileStr.split('_')[0])
        vectUnderTest = img2vector('./testDigits/%s' %fileNameStr)
        classifierResult = knn.classifyy0(vectUnderTest,trainingMat,hwLabels,k=3)
        if classifierResult != numStr:
            errorCount += 1
        print("kNN预测结果： %s -- 实际结果： %s" %(classifierResult,numStr))
    print("kNN失误率为：%s" %(errorCount/mTest))


def main():

    imgVector = img2vector(filename='./trainingDigits/0_0.txt')
    handwriteClassifier()


if __name__ == '__main__':
    main()