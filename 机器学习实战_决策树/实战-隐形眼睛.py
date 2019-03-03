# -*- coding: utf-8 -*-
'''
作者:     李高俊
    版本:     1.0
    日期:     2018/11/22/
    项目名称： 预测隐形眼睛类型
'''
import numpy as np
import decisiontree_practice as tree_pra
import tree_plotter as tree_plt
"""
步骤：
    1. 收集数据：已有数据集lense.txt
    2. 准备数据：解析数据集，将其转换为能被程序理解的数据
    3. 分析数据：快速检查数据，看看有无错误数据，使用createplot()绘制树形图
    4. 训练模型：creattree函数
    5. 测试模型：编写测试函数判断函数验证决策树的正误
    6. 使用模型：储存树的结构，以便下次使用
"""
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = tree_pra.creatTree(lenses,lensesLabels)
tree_plt.createPlot(lensesTree)