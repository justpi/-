# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018///
    项目名称： 从个人广告中获取区域倾向
'''
import numpy as np
import os
import feedparser
import 垃圾邮件分类 as gab
import bayes_practice as bay
import random
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortFreq[:30]


def localWords(feed0,feed1):
    docList = []
    classList = []
    fulltext = []
    minlen = min(len(feed0['entries']),len(feed1['entries']))
    for i in range(minlen):
        wordList = gab.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fulltext.extend(wordList)
        classList.append(1)
        wordList = gab.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fulltext.extend(wordList)
        classList.append(0)
    vocabList = bay.createVocabList(docList)
    # top30words = calcMostFreq(vocabList,fulltext)
    # print(top30words)
    # for word in top30words:
    #     if word in vocabList:
    #         vocabList.remove(vocabList[word])
    trainingSet = list(range(2*minlen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(1,len(trainingSet)))
        testSet.append(randIndex)
        trainingSet.remove(trainingSet[randIndex])
    trainMat = []
    trainClass = []
    for i in trainingSet:
        trainMat.append(bay.setOfWord2Vec(vocabList,docList[i]))
        trainClass.append(classList[i])
    p0V,p1V,pSpam = bay.trainNB0(trainMat,trainClass)
    errorCount = 0.0
    for i in testSet:
        wordVector = bay.setOfWord2Vec(vocabList,docList[i])
        if bay.classifyNB(wordVector,p0V,p1V,pSpam) != classList[i]:
            print(bay.classifyNB(wordVector,p0V,p1V,pSpam),"----",classList[i])
            errorCount += 1
        return errorCount/len(testSet)

def feed_parser(url):
    return feedparser.parse(url)
def main():
    error = localWords(feed_parser('http://www.chinadaily.com.cn/rss/sports_rss.xml'),feed_parser('http://www.chinadaily.com.cn/rss/entertainment_rss.xml'))
    print(error)


if __name__ == '__main__':
    main()