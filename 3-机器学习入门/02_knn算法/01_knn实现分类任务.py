# -*- coding: utf-8 -*-
# @Time    : 2026/5/26 07:20
# @Author  : WanLi
# @File    : 01_knn实现分类任务.py

# 分类流程
# 1. 计算未知样本与所有训练样本之间的距离
# 2. 按照距离从小到大排序（距离越近，相似度越高）
# 3. 选取距离最近的 K 个邻居样本
# 4. 统计 K 个邻居中各类别出现的次数（投票机制）
# 5. 将出现次数最多的类别作为预测结果
# 6. 若出现票数相同（平票），则按照算法规则进行处理（如选择距离更近或默认类别：默认选择类别值较小的那个label）

# 1、导包，此处导入knn分类模型
from sklearn.neighbors import KNeighborsClassifier

# 2.准备数据(此处我们模拟数据)
# 先模拟特征数据
x_train = [ [0],[1],[2],[3],[4] ]
x_test = [ [5] ]

# 在模拟标签数据y (假设：1：垃圾邮件，0:正常邮件)
y_train = [ 0,0,0,1,0 ]  #此时0和1是标签索引
# TODO 3.预测x_test中的5是否为垃圾邮件
# TODO 3.1 创建分类模型
knn_model = KNeighborsClassifier(n_neighbors=3)  #邻居偏移量
# TODO 3.2 训练模型
knn_model.fit(x_train,y_train)
# TODO 3.3 预测
y_predict = knn_model.predict(x_test)
print(f'预测结果为：{y_predict}')
