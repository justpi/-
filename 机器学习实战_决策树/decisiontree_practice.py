# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018/11/20/
    项目名称： 决策树模型--ID3算法实现
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator
"""
步骤：
    1. 收集数据：任意方法
    2. 准备数据：树结构只适用于离散化的数据，对于数值型的数据，要进行离散化
    3. 分析数据：可使用任何方法，构造完成树之后，应该检查图形是否符合预期
    4. 训练算法：构造树的数据结构
    5. 测试算法：使用经验树计算错误率
    6. 使用算法：适用于任意分类问题，帮助理解数据内在含义
"""
def calcShanonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for feaVect in dataset:
        currentLabel = feaVect[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shanonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries)
        shanonEnt -= prob * math.log(prob,2)
    return shanonEnt


def creatdataset():
    dataset = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataset,labels


def splitData(dataset,axis,value):
    retDataset = []
    for featVector in dataset:
        if featVector[axis] == value:
            reduceFeatVec = featVector[:axis]
            reduceFeatVec.extend(featVector[axis+1:])
            retDataset.append(reduceFeatVec)
    return retDataset


def chooseBestFeaturetoSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseStropy = calcShanonEnt(dataset)
    bestinfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataset]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDateset = splitData(dataset,i,value)
            prob = len(subDateset)/float(len(dataset))
            newEntropy += prob * calcShanonEnt(subDateset)
        infoGain = baseStropy - newEntropy
        if infoGain > bestinfoGain:
            bestinfoGain = infoGain
            bestFeature = i
        return bestFeature


def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(classCount.items(),operator.itemgetter(1),reverse=True)
    return sortedclassCount[0][0]


def creatTree(dataset, labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classlist)
    bestFeature = chooseBestFeaturetoSplit(dataset)
    bestFeatLabel = labels[bestFeature]
    mytree = {bestFeatLabel:{}}
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataset]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        sublabels = labels[:]
        mytree[bestFeatLabel][ value] = creatTree(splitData(dataset,bestFeature,value),sublabels)
    return mytree
def getnumLeafs(mytree):
    numLeafs = 0
    firstStr = list(mytree.keys())[0]
    secondStr = mytree[firstStr]
    for key in secondStr.keys():
        if type(secondStr[key]).__name__ == 'dict':
            numLeafs += getnumLeafs(secondStr[key])
        numLeafs += 1
    return numLeafs


def getTreedpth(mytree):
    maxDepth = 0
    firstStr = list(mytree.keys())[0]
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ =='dict':
            thisDepth = 1 + getTreedpth(secondDict[key])
        else :
            thisDepth =1
        if thisDepth >maxDepth :
            maxDepth = thisDepth
    return maxDepth


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

def main():
    dataset,labels = creatdataset()
    shanonEnt = calcShanonEnt(dataset=dataset)
    retdataset = splitData(dataset,axis=0,value=0)
    mytree = creatTree(dataset,labels)
    numLeafs = getnumLeafs(mytree)
    numDepth = getTreedpth(mytree)
    storeTree(mytree,'myTree.txt')
    print(grabTree('myTree.txt'))
if __name__ == '__main__':
    main()