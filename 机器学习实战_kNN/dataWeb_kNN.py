# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018///
    项目名称： 约会网站
'''
"""
步骤：
    1.收集数据：从datingTestSet2文本中提取数据
    2.准备数据：使用python解析文本数据
    3.分析数据：用matplotlab可视化分析文本数据
    4.训练算法：不适用
    5.测试算法：使用一部分数据作为测试集进行测试
    6.使用算法：实际运行
"""
import numpy as np
import pandas as pd
import knn_practice as knn
import matplotlib.pyplot as plt
def file2matrix(filename):
    fr = open(filename)
    arrayOLine = fr.readlines()
    returnMat = np.zeros((len(arrayOLine),3))
    classLabelVector = []
    index = 0
    for line in arrayOLine:
        line = line.strip()
        listline = line.split('\t')
        returnMat[index,:] = listline[0:3]
        classLabelVector.append(int(listline[-1]))
        index += 1
    return returnMat,classLabelVector


def datingclassTest():
    hoRitio = 0.1
    datingmat,labels = file2matrix('./datingTestSet2.txt')
    normMat,ranges_mat,minvals = knn.autoNorm(datingmat)
    m = datingmat.shape[0]
    num_test = int(m * hoRitio)
    errorCount = 0
    for i in range(num_test):
        classifyresult = knn.classifyy0(normMat[i,:],normMat[num_test:m,:],labels[num_test:m],3)
        print("kNN测试结果：%s--实际结果：%s ." %(classifyresult,labels[i]))
        if classifyresult != labels[i]:
            errorCount += 1
    print("分类失误率为： %s" %(errorCount/num_test))
    return


def classifyPerson():
    resultList = ['一点都不喜欢','一般喜欢','特别喜欢']
    person_tats = float(input("他每周玩视频游戏的时间百分比："))
    person_ffMiles = float(input('他每年获得的飞行常客里程数：'))
    person_icecream = float(input("他每周消费的冰淇淋公升数："))
    datingMat,datinglabels = file2matrix('./datingTestSet2.txt')
    normMat,norm_ranges,norm_minvals = knn.autoNorm(datingMat)
    person_date = [person_ffMiles,person_tats,person_icecream]
    norm_person = (person_date - norm_minvals)/ norm_ranges
    classifyresult = knn.classifyy0(inX=norm_person,dataset=normMat,labels=datinglabels,k=3)
    print("你可能喜欢这个男孩： %s" %resultList[classifyresult-1])


def main():
    classifyPerson()
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.scatter(dataset[:,1],dataset[:,2],10.0*np.array(labels),10.0*np.array(labels))
    # plt.xlabel('game')
    # plt.ylabel('ice cream')
    # plt.title('ice cream -- game')
    # ax2 = fig.add_subplot(122)
    # ax2.scatter(dataset[:,0],dataset[:,1],10.0*np.array(labels),10.0*np.array(labels))
    # plt.xlabel('airport')
    # plt.ylabel('game')
    # plt.title('airport -- game')
    # plt.show()



if __name__ == '__main__':
    main()