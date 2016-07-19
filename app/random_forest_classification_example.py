#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: GBK -*-

"""
Random Forest Classification Example.
"""
from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

if __name__ == "__main__":
    sc = SparkContext(appName="PythonRandomForestClassificationExample")

    # 加载和解析数据文件为RDD
    dataPath = "/home/zhb/Desktop/work/DecisionTreeShareProject/app/sample_libsvm_data.txt"
    print(dataPath)

    data = MLUtils.loadLibSVMFile(sc,dataPath)
    # 将数据集分割为训练数据集和测试数据集
    (trainingData,testData) = data.randomSplit([0.7,0.3])
    print("train data count: " + str(trainingData.count()))
    print("test data count : " + str(testData.count()))

    # 训练随机森林分类器
    # categoricalFeaturesInfo 为空，表示所有的特征均为连续值
    # 实践中使用更多numTrees
    # 设置featureSubsetStrategy="auto"，让算法来选择
    model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                         numTrees=3, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)

    # 测试数据集上预测
    predictions = model.predict(testData.map(lambda x: x.features))
    # 打包真实值与预测值
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    # 统计预测错误的样本的频率
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
    print('Random Forest Test Error = %5.3f%%'%(testErr*100))
    print("RandomForest Learned classifiction tree model : ")
    print(model.toDebugString())

    # 保存和加载训练好的模型
    modelPath = "/home/zhb/Desktop/work/DecisionTreeShareProject/app/myRandomForestClassificationModel"
    model.save(sc, modelPath)
    sameModel = RandomForestModel.load(sc, modelPath)
