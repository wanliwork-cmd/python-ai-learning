# -*- coding: utf-8 -*-
# @Time    : 2026/5/28 13:00
# @Author  : WanLi
# @File    : Kmeans算法API应用.py

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 1.准备数据特征2列标签1列
#参数：n_samples：样本数，n_features：特征数，centers：中心点数，cluster_std：簇标准差，random_state：随机种子
X, y = make_blobs(
    n_samples=1000,
    n_features=2,
    #centers=2, # 中心点数 可以自定义坐标
    #cluster_std=1.0, # 簇标准差 可以自定义
    centers=[ [-1,-1],[0,0],[1,1],[2,2]],
    cluster_std=[0.4, 0.2, 0.2, 0.25]
)

# print(X.shape, y.shape)
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# 模型使用
model = KMeans(n_clusters=4)  # 创建KMeans模型，指定簇数为4
y_pred = model.fit_predict(X)  # 模型训练并预测

#可视化 按照预测值，通过颜色划分类别，绘制散点图看看数据分布
#参数： x：特征值，y：目标值，c：颜色，s：点的大小，marker：点的形状
plt.scatter(X[:, 0], X[:, 1], c=y_pred) #y_pred 有几类类别，就显示几类颜色 n_clusters=4 就是4类颜色
plt.show()

