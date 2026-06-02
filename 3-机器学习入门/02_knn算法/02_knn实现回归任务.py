# -*- coding: utf-8 -*-
# @Time    : 2026/5/26 07:20
# @Author  : WanLi
# @File    : 01_knn实现回归任务.py

# 回归流程
# 1. 计算未知样本与每一个训练样本之间的距离
# 2. 按照距离从小到大排序（距离越近，相似度越高）升序排序
# 3. 选取距离最近的 K 个邻居样本
# 4. 统计 K 个邻居对应的目标值（数值结果）
# 5. 对 K 个邻居的目标值进行平均（或加权平均）
# 6. 将计算后的结果作为最终预测值


# 1、导包，此处导入knn回归模型
from sklearn.neighbors import KNeighborsRegressor

# 2.准备数据(此处我们模拟数据)
# 先模拟特征数据
x_train = [ [0],[1],[2],[3],[4] ]
x_test = [ [5] ]

# 在模拟标签数据y (计算平均值)
y_train = [ 70,80,100,110,120 ]
# TODO 3.预测x_test中的5的房价
# TODO 3.1 创建回归模型
knn_model = KNeighborsRegressor(n_neighbors=3)  #邻居偏移量
# TODO 3.2 训练模型
knn_model.fit(x_train,y_train)
# TODO 3.3 预测
y_predict = knn_model.predict(x_test)
print(f'预测结果为：{y_predict}')
