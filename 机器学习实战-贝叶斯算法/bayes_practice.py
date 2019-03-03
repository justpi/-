# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018/11/22/
    项目名称： 朴素贝叶斯算法
'''
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
def loaddataSet():
    postingList = [
        ['my','dog','has','flea','problem','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("当前字典中无%s" %word)
    return returnVec
def numaricList(inputSet):
    vocabList = createVocabList(inputSet)
    numDataSet = []
    for word in inputSet:
        numDataSet.append(setOfWord2Vec(vocabList,word))
    return numDataSet


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)     #p(c1)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p1Denom = 2.0
    p0Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect,p1Vect,pAbusive
def classifyNB(vect2Classify,p0Vect,p1Vect,pClass1):
    p1 = sum(vect2Classify * p1Vect) + np.log(pClass1)
    p0 = sum(vect2Classify * p0Vect) + np.log(1 - pClass1)
    if p1 > p0:
        return 1,p1
    else:
        return 0,p0



def main():
    dataSet,labels = loaddataSet()
    vocabList = createVocabList(dataSet)
    print(vocabList)
    print('_'*32)
    numDataSet = numaricList(dataSet)
    p0Vect,p1Vect,pAbusive = trainNB0(numDataSet,labels)
    print(classifyNB(numDataSet[0],p0Vect,p1Vect,pAbusive))
if __name__ == '__main__':
    main()