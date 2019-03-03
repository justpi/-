# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018/11/21/
    项目名称： 画出决策树的图
'''
import decisiontree_practice as tree_pra
import matplotlib.pyplot as plt


decisionNode = dict(boxstyle='sawtooth',fc = '0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_arg = dict(arrowstyle='<-')
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords = 'axes fraction',xytext = centerPt,
                           textcoords = 'axes fraction',va= 'center',ha='center',bbox=nodeType,
                           arrowprops = arrow_arg)
def creatPlot():
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    creatPlot.ax1 = plt.subplot(111,frameon = False)
    plotNode('decsionNode',(0.5,0.1),(0.3,0.5),decisionNode)
    plotNode('leafNode',(0.8,0.3),(0.3,0.8),leafNode)
    plt.show()
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2 + cntrPt[1]
    createPlot.ax1.text(xMid,yMid,txtString) #标记节点的属性值


def plottree(mytree,parentPt,nodeTxt):
    numLeafs = tree_pra.getnumLeafs(mytree) #计算决策树宽
    numDepth = tree_pra.getTreedpth(mytree) #计算决策树高
    firstStr = list(mytree.keys())[0]
    cntrPt = (plottree.xOff + (1.0 + float(numLeafs))/2.0/plottree.totalW,plottree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = mytree[firstStr]
    plottree.yOff = plottree.yOff - 1.0/plottree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
           plottree(secondDict[key], parentPt, str(key))
        else:
            plottree.xOff = plottree.xOff + 1.0/plottree.totalW
            plotNode(secondDict[key],(plottree.xOff,plottree.yOff),cntrPt,leafNode)
            plotMidText((plottree.xOff,plottree.yOff),cntrPt,str(key))
    plottree.yOff = plottree.yOff + 1.0/plottree.totalD


def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks = [],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon = False, **axprops)
    plottree.totalW = float(tree_pra.getnumLeafs(inTree))   #储存树的宽度
    plottree.totalD = float(tree_pra.getTreedpth(inTree))   #储存树的深度
    plottree.xOff = -0.5/plottree.totalW
    plottree.yOff = 1.0
    plottree(inTree,(0.5,1.0),"")
    plt.show()
