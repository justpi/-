# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018///
    项目名称： adaboost算法
'''
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import tensorflow
"""
    1. 收集数据：
    2. 准备数据：
    3. 分析数据：
    4. 训练算法：
    5. 测试算法：
    6. 使用算法：
"""
def loadSimpData():
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1.3, 1.],
                         [1., 1.],
                         [2., 1.]])
    classLabel = np.mat([1.0,1.0,-1.0,-1.0,1.0]).transpose()
    return dataMat,classLabel





def main():
    dataMat,classLabel = loadSimpData()



if __name__ == '__main__':
    main()