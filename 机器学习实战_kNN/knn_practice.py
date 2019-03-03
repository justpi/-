# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018///
    项目名称： K-近邻算法手动实现
'''
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
def creat_train_dataset():
    train_set = pd.DataFrame()
    train_set['movie name'] = ['California Man',"He's Not Really into Dudes","beautiful Woman",'Kevin Longblade',
                               'Robo Slayer 3000',"Amped II"]
    train_set['fight shot'] = [3,2,1,101,99,98]
    train_set['kiss shot'] = [104,100,81,10,5,2]
    train_set['type of movie'] = ["love",'love','love','action','action','action']
    return train_set


def creat_dataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


def classifyy0(inX,dataset,labels,k):
    dataSetSize = dataset.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1)) - dataset
    sqdiffMat = diffMat ** 2
    sqDistence = np.sum(sqdiffMat,axis=1)
    distence = sqDistence ** 0.5
    sortDistence = np.argsort(distence)
    countcalss = {}
    for i in range(k):
        voteIlabel = labels[sortDistence[i]]
        countcalss[voteIlabel] = countcalss.get(voteIlabel,0) + 1
    sortedcountclass = sorted(countcalss.items(),key=operator.itemgetter(1),reverse=True)
    return sortedcountclass[0][0]

def autoNorm(dataset):
    minvals = dataset.min(0)
    maxvals = dataset.max(0)
    ranges = maxvals-minvals
    normdataset = np.zeros(shape=np.shape(dataset))
    m = dataset.shape[0]
    normdataset = dataset - np.tile(minvals,(m,1))
    normdataset = normdataset/(np.tile(ranges,(m,1)))
    return normdataset,ranges,minvals

