# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018///
    项目名称： 逻辑回归
'''
import numpy as np
import os
import pandas as pd
import math
import matplotlib.pyplot as plt

def loadData():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt',encoding='ISO-8859-1')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):

    return 1.0/(1 + np.exp(-inX))


def gradeAscent(dataMat,labelMat,maxCycle,alpha):   #批梯度下降
    dataMatrix = np.mat(dataMat)
    labelVector = np.mat(labelMat).transpose()
    m,n = dataMatrix.shape
    weight = np.ones((n,1))
    for i in range(maxCycle):
        label_calc = dataMatrix * weight
        h = sigmoid(label_calc)
        error = (labelVector - h)
        weight = weight + alpha * dataMatrix.transpose() * error
    return weight


def stocGradAscent0(dataMat,labelMat,alpha,weight):    #随机梯度下降
    m,n = dataMat.shape
    for i in range(m):

        h = sigmoid(sum(dataMat[i] * weight))
        error = labelMat[i] - h
        weight = weight + alpha * error * dataMat[i]
    return weight


def stocGradAscent1(dataMat,labelMat,numIter):  #改进后的随即梯度下降函数
    import random
    m,n = dataMat.shape
    weight = np.ones(n)
    for i in range(numIter):
        for j in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = random.uniform(0,m)
            h = sigmoid(inX = (sum(dataMat[j] * weight)))
            error = labelMat[j] - h
            weight = weight + alpha * error * dataMat[j]
    return weight

def plotBestFit(weight):
    dataMat,labels = loadData()
    dataMat = np.array(dataMat)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    m = dataMat.shape[0]
    for i in range(m):
        if int(labels[i]) == 1:
            xcord1.append(dataMat[i,1])
            ycord1.append(dataMat[i,2])
        else:
            xcord2.append(dataMat[i,1])
            ycord2.append(dataMat[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s = 30,c = 'red',marker='s')
    ax.scatter(xcord2,ycord2,s = 30,c = 'green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weight[0] - weight[1] * x)/weight[2]
    ax.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()



def main():
    dataSet,labels = loadData()
    # weight = gradeAscent(dataSet,labels,500,0.001)
    # weight = np.ones(np.array(dataSet).shape[1])
    # x0cord = []
    # x1cord = []
    # x2cord = []
    # for i in range(400):
    #     x0cord.append(weight[0])
    #     x1cord.append(weight[1])
    #     x2cord.append(weight[2])
    #     weight = stocGradAscent0(np.array(dataSet),labels,0.01,weight)
    # x = list(range(400))
    # plt.plot(x,x0cord)
    # plt.ylabel("X0")
    # plt.show()
    # plt.plot(x,x1cord)
    # plt.ylabel("X1")
    # plt.show()
    # plt.plot(x,x2cord)
    # plt.ylabel("X2")
    # plt.show()
    for i in  range(20):
        weight = stocGradAscent1(np.array(dataSet),labels,i)
        plotBestFit(np.array(weight))

if __name__ == '__main__':
    main()