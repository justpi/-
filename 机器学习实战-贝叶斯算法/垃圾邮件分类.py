# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018///
    项目名称： 垃圾邮件分类
'''
import numpy as np
import matplotlib.pyplot as plt
import bayes_practice as by
import os
import random
"""
步骤：
    1.收集数据：已有数据
    2.准备数据：将文本数据转化为矩阵
    3.分析数据：检查确保词条准确性
    4.训练算法：使用trainNB0()函数训练算法
    5.测试算法：使用classifyNB()函数
    6.使用算法：构建一个完整的程序
"""
def textParse(bigString):
    import re
    listOfToken = re.split(r'\W*',bigString)
    return [token.lower() for token in listOfToken if len(token) > 0]


def spamText():
    from io import StringIO
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('./email/spam/%d.txt' % i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('./email/ham/%d.txt' %i,encoding='ISO-8859-1').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = by.createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(1,len(trainingSet)))
        testSet.append(randIndex)
        del(trainingSet[randIndex])
    trainingMat = []
    trainingclass = []
    for i in trainingSet:
        trainingMat.append(by.setOfWord2Vec(vocabList,docList[i]))
        trainingclass.append(classList[i])
    p0V,p1V,pSpam = by.trainNB0(trainingMat,trainingclass)
    errorCount = 0.0
    for i in testSet:
        wordVector = by.setOfWord2Vec(vocabList,docList[i])
        if by.classifyNB(wordVector,p0V,p1V,pSpam) != classList[i]:
            errorCount += 1
        return errorCount/len(testSet)




def main():
    print(spamText())


if __name__ == '__main__':
    main()
